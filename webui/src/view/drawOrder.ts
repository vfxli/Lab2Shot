/** The 3D stage's draw order (three's renderOrder: one sort across every object of a list, opaque then transparent).
 * Every part takes its number from here, so the bands never overlap: the plate first, bodies, the ground grid, lines
 * drawn over the scene, and last the skeletons drawn over everything (overlay), which alone own overlay*: their depth
 * clear comes after everything else has been drawn (grid and lines included), then all their bones, then all their
 * joint balls. */
export const ORDER = {
  plate: -10, // the image plane behind the scene when looking through a camera
  body: 1, // a mesh's wireframe over its own surface (a model's, a skinned character's); the surfaces themselves stay at 0
  grid: 5, // the ground grid: after bodies (it does not hide what is below the ground), before lines over the scene
  lines: 6, // lines drawn over the scene without an order of their own (lines3d.tsx FatLines `overlay`: the picked box): over the grid too
  overlayClear: 9, // the overlay skeletons' one depth clear (elements3d.tsx overlayClear)
  overlayBones: 10,
  overlayBalls: 11,
} as const;
