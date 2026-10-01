// Point cloud shaders used by view/points3d.tsx: shared GLSL and disc rendering (gl_Points, round with soft edges).
//
// Points are drawn only as discs, colored either by their own color or by a single tint. Size is specified only in
// screen pixels (`uSize`), which maps directly to gl_PointSize.

export const COLOR_MODES = { color: 0, constant: 1 } as const;

const COMMON = /* glsl */ `
uniform int uColor;
uniform vec3 uTint;
vec3 toLinear(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
#ifdef GRID
uniform highp sampler2D uDepth;
uniform highp sampler2D uRgb; // alpha marks pixels that are points
uniform float uHasRgb;
uniform vec3 uGridTint;
uniform vec3 uRamp; // (near, far, on): a distance map's colour ramp (server/view_data.py _distance_grid_view), warm near, cool far
uniform vec3 uRampNear; // the ramp's colour at v = 0 and its change per unit v, as the server gives them (grid ramp_colour:
uniform vec3 uRampSlope; // view_data.py RAMP_NEAR / RAMP_SLOPE, the one definition)
uniform vec4 uGrid; // grid width, step, picture width, picture height
uniform float uFocal;
uniform vec2 uPrincipal; // (cx, cy) in pixels: the camera's principal point (the image centre unless the solver provides one)
uniform mat4 uCam;
bool gridPoint(int k, inout vec3 p, inout vec3 rgb) {
  int gw = int(uGrid.x);
  ivec2 t = ivec2(k % gw, k / gw);
  vec4 c4 = texelFetch(uRgb, t, 0);
  if (c4.a < 0.5) return false;
  float z = texelFetch(uDepth, t, 0).r;
  float c = float(t.x) * uGrid.y;
  float r = float(t.y) * uGrid.y;
  float x = (c + 0.5 - uPrincipal.x) / uFocal * z;
  float y = (r + 0.5 - uPrincipal.y) / uFocal * z;
  p = mat3(uCam) * vec3(x, -y, -z) + uCam[3].xyz;
  float v = (z - uRamp.x) / max(uRamp.y - uRamp.x, 1e-6);
  rgb = uRamp.z > 0.5 ? clamp(uRampNear + v * uRampSlope, 0.0, 1.0) : uHasRgb > 0.5 ? c4.rgb : uGridTint;
  return true;
}
#endif
vec3 pointColor(vec3 rgb) {
  if (uColor == 0) return toLinear(rgb);
  return uTint;
}
`;

export const POINTS_VERT = /* glsl */ `
${COMMON}
attribute vec3 rgb;
uniform float uSize;
uniform float uPixelRatio;
varying vec3 vColor;
varying float vSize;
void main() {
#ifdef GRID
  vec3 pos = vec3(0.0);
  vec3 col = vec3(0.0);
  if (!gridPoint(gl_VertexID, pos, col)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 1.0; vSize = 1.0; vColor = col; return; }
#else
  vec3 pos = position;
  vec3 col = rgb;
#endif
  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp(uSize * uPixelRatio, 1.0, 256.0);
  vSize = gl_PointSize;
  vColor = pointColor(col);
}`;

export const POINTS_FRAG = /* glsl */ `
varying vec3 vColor;
varying float vSize;
void main() {
  float r = length(2.0 * gl_PointCoord - 1.0);
  float edge = min(0.5, 2.0 / max(vSize, 1.0));
  float alpha = 1.0 - smoothstep(1.0 - edge, 1.0, r);
  if (alpha <= 0.02) discard;
  gl_FragColor = vec4(vColor, alpha);
  #include <colorspace_fragment>
}`;
