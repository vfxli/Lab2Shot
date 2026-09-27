import * as THREE from "three";
import { columnMajor } from "../model/viewFormat";

/** A 4x4 of rows `rows` at `i` (clamped to the samples there are) as a three.js matrix. */
export function matrixAt(rows: Float32Array, i: number): THREE.Matrix4 {
  const n = rows.length / 16;
  return new THREE.Matrix4().fromArray(columnMajor(rows, Math.min(i, n - 1), 4));
}
