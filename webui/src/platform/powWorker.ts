/// <reference lib="webworker" />
import { solve } from "./pow";
import { workerAnswers } from "./work";

/** The registration page's proof of work (pow.ts), off the page's thread: the form stays responsive while it runs. */
workerAnswers<{ nonce: string; bits: number }>(({ nonce, bits }) => ({ answer: solve(nonce, bits) }));
