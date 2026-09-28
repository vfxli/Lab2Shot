"""HTTPS for users on other machines. Browsers let a page read and write the user's folders (the file browser) only
over HTTPS, or on the machine itself (localhost).

With HTTPS on (`lab2shot ui --https`, or the setting server.https) the server makes, once, a certificate authority of this server (work/tls/ca.pem) and from it a certificate
for every name and address the server has. Each user machine installs ca.pem once (the page /api/tls/ca.pem hands it
out), after that the browser trusts the server like any website. A new address (another
network) gets a new server certificate from the same authority, so users never install anything again.

The authority may only vouch for this server (NameConstraints, critical): its host names and the names of setting
server.names, the private and loopback address ranges (a machine moving between networks keeps its authority), and
this machine's other addresses one by one. Whoever took its key (work/tls/ca.key, readable by this account only)
could mint certificates for those names and nothing else: not for any other website the users' machines visit. A name
or public address outside what the authority permits (server.names changed, say) makes a new authority, and each user
machine installs the new ca.pem once (the server log says so).
"""

from __future__ import annotations

import datetime
import ipaddress
import json
import os
import socket
import subprocess
from pathlib import Path

from ..config import settings

CA_DAYS = 3650
SERVER_DAYS = 800
# address ranges the authority may vouch for whatever the machine's address in them: loopback, private (RFC 1918, ULA),
# link-local and the shared range VPNs such as Tailscale hand out
PRIVATE_NETS = ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "169.254.0.0/16",
                "::1/128", "fc00::/7", "fe80::/10")


def folder() -> Path:
    return settings().work_dir / "tls"


def own_addresses() -> set[str]:
    """This machine's own IP addresses: loopback, and those of its network interfaces (`hostname -I`)."""
    ips = {"127.0.0.1", "::1"}
    try:
        ips |= set(subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout.split())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ips


def _named(listed: object) -> tuple[set[str], set[str]]:
    """(host names, IP addresses) of `listed` (setting server.names), and this machine's own host names."""
    host = socket.gethostname().lower()
    hosts, ips = {"localhost", host, f"{host}.local"}, set()
    for name in str(listed).replace("，", ",").replace(",", " ").split():
        try:
            ips.add(str(ipaddress.ip_address(name)))
        except ValueError:
            hosts.add(name.lower())
    return hosts, ips


def host_names() -> set[str]:
    """The host names the server answers to (no addresses: those need no lookup of the machine's interfaces), as
    server.names is saved now: a name added there is answered at once (the certificate waits for the restart)."""
    return _named(settings().value("server.names"))[0]


def names() -> tuple[list[str], list[str]]:
    """(host names, IP addresses) the server answers to: its own, and the ones the administrator named for access from
    outside (setting server.names: a tunnel's domain)."""
    hosts, ips = _named(settings()["server.names"])
    return sorted(hosts), sorted(ips | own_addresses())


def _key():
    from cryptography.hazmat.primitives.asymmetric import ec

    return ec.generate_private_key(ec.SECP256R1())


def _write_key(path: Path, key) -> None:
    """A private key, readable by this account only from the moment it exists: the file is created with its
    permissions (O_EXCL, 0600), never written first and narrowed after, in a folder only this account may enter."""
    from cryptography.hazmat.primitives import serialization

    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.unlink(missing_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))


def _load_key(path: Path):
    from cryptography.hazmat.primitives import serialization

    return serialization.load_pem_private_key(path.read_bytes(), None)


def _permitted(hosts: list[str], ips: list[str]):
    """The authority's NameConstraints for these names and addresses: each host name (and what lies under it), the
    PRIVATE_NETS, and every other address on its own."""
    from cryptography import x509

    nets = [ipaddress.ip_network(n) for n in PRIVATE_NETS]
    alone = [ipaddress.ip_network(i) for i in ips if not any(ipaddress.ip_address(i) in n for n in nets)]
    return x509.NameConstraints(permitted_subtrees=[x509.DNSName(h) for h in hosts] + [x509.IPAddress(n) for n in nets + alone],
                                excluded_subtrees=None)


def _covers(ca, hosts: list[str], ips: list[str]) -> bool:
    """Does the authority's NameConstraints permit every one of these names and addresses (False: it has none)?"""
    from cryptography import x509

    try:
        permitted = ca.extensions.get_extension_for_class(x509.NameConstraints).value.permitted_subtrees or []
    except x509.ExtensionNotFound:
        return False
    names = [g.value.lower() for g in permitted if isinstance(g, x509.DNSName)]
    nets = [g.value for g in permitted if isinstance(g, x509.IPAddress)]
    return (all(any(h == n or h.endswith(f".{n}") for n in names) for h in hosts)
            and all(any(ipaddress.ip_address(i) in n for n in nets) for i in ips))


def _authority(hosts: list[str], ips: list[str]):
    """This server's certificate authority, permitted these names and addresses: kept while it permits them, else made
    anew (and the server log says each user machine installs the new ca.pem)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.x509.oid import NameOID

    from .. import logs
    from ..messages import Msg

    key_file, cert_file = folder() / "ca.key", folder() / "ca.pem"
    if key_file.exists() and cert_file.exists():
        cert = x509.load_pem_x509_certificate(cert_file.read_bytes())
        if _covers(cert, hosts, ips):
            return _load_key(key_file), cert
        logs.say(logs.get("server"), Msg("W-TLS-NEWAUTHORITY", names=" ".join(hosts + ips)))
    folder().mkdir(parents=True, exist_ok=True)
    key = _key()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"Lab2Shot {socket.gethostname()}"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Lab2Shot")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=CA_DAYS))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                         key_encipherment=False, data_encipherment=False, key_agreement=False,
                                         encipher_only=False, decipher_only=False), critical=True)
            .add_extension(_permitted(hosts, ips), critical=True)  # RFC 5280 4.2.1.10: a CA marks it critical
            # a CA states its key identifier (RFC 5280 4.2.1.2): the server certificate points back to it. Without it
            # OpenSSL 3's default verification refuses the chain ("Missing Authority Key Identifier"): browsers only
            # warn, but Python clients (the command line, DCC plugins) cannot connect at all.
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, hashes.SHA256()))
    _write_key(key_file, key)
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return key, cert


def ensure() -> tuple[Path, Path]:
    """(certificate, key) for the server, valid for its current names and addresses."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    hosts, ips = names()
    cert_file, key_file, made_for = folder() / "server.pem", folder() / "server.key", folder() / "server.json"
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key, ca = _authority(hosts, ips)
    wanted = {"hosts": hosts, "ips": ips, "authority": ca.serial_number}  # a new authority: a new server certificate too
    if cert_file.exists() and key_file.exists() and made_for.exists():
        current = x509.load_pem_x509_certificate(cert_file.read_bytes())
        fresh = current.not_valid_after_utc - now > datetime.timedelta(days=30)
        if fresh and json.loads(made_for.read_text()) == wanted:
            return cert_file, key_file
    key = _key()
    alt = [x509.DNSName(h) for h in hosts] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])]))
            .issuer_name(ca.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=SERVER_DAYS))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    _write_key(key_file, key)
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM))
    made_for.write_text(json.dumps(wanted))
    return cert_file, key_file


def authority_file() -> Path | None:
    """The certificate users install, if HTTPS was ever turned on."""
    f = folder() / "ca.pem"
    return f if f.exists() else None
