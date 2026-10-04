# Lab2Shot Maya Plug-in: Installation

[中文](INSTALL.md) | **English**

For Maya 2024, 2025 and 2026 (Windows). The plug-in uses only Maya's own Python and Qt: nothing is installed into Maya,
and Maya's installation folder is not changed.

## Install

1. Unzip this package. It holds a `lab2shot.mod` file and a `lab2shot` folder.
2. Copy both into `Documents\maya\modules\` (that is `C:\Users\<your user name>\Documents\maya\modules\`; create the
   `modules` folder if there is none). Afterwards it looks like this:
   ```
   Documents\maya\modules\lab2shot.mod
   Documents\maya\modules\lab2shot\scripts\...
   ```
3. Restart Maya. When a "Lab2Shot" menu appears in the menu bar, it is installed. On the first start Maya may ask
   whether to allow `userSetup.py` to run: allow it.

Upgrade: copy the two items from the new package over the old ones and restart Maya. Uninstall: delete the two items
and restart Maya.

## First Steps

1. **Lab2Shot > Log In…**: enter the server address (the URL your administrator gives you, for example
   `https://lab2shot.example.com`), your user name and password. No account yet? Sign up on the web page.
2. **Lab2Shot > Open Panel**. The panel opens on All Tools: a search field at the top; once something is selected in
   the scene (a camera, a camera with an image plane, a character, a skeleton), **For the Selection** lists the tools
   that take it; below, the tools by category in tabs (**Recent** last). Each card shows the tool's name, one line of
   what it does, and what it needs ("Needs: Image, Camera").
3. Click a card: a Lab2Shot node (`lab2shot1`) is added to the scene and the panel shows that tool. A card from
   **For the Selection** also binds what is selected to the inputs that take it. When the current node already has a
   tool, clicking a card makes a new node: one node per task, and earlier result versions stay as they are.
4. The tool page has three steps, top to bottom:
   - **1 Inputs**: one card per input. **Use Selected** takes what is selected in the scene; **Choose File** /
     **Choose Sequence** picks from disk (select every frame of a sequence in its folder, or just one of its frames).
     A sequence shows its first frame, last frame and frame count.
   - **2 Settings**: the common settings are shown; **Advanced** is folded at first. Values marked **From Scene** were
     read from the scene (frame range, frame rate, focal length…) and can be changed.
   - **3 Cook**: click **Cook**. The work runs on the server and Maya stays usable; the progress shows under the
     button, which becomes **Cancel** — you can cancel at any time.
5. When it is done, the results are imported as new objects under the `Lab2Shot` group in the Outliner
   (`lab2shot1_v001`), in the namespace `l2s_lab2shot1_v001`. Every cook gives a new version; the plug-in never changes
   or deletes anything that was in the scene. An import is one undo step: one Ctrl+Z removes all of it.
   When the results hold a camera, a new image plane is hung on the new camera: the bound picture when one was bound,
   otherwise the picture on the bound camera's own image plane; with neither, none is hung and the panel says so.
6. The result versions are in a row at the bottom of the tool page. Click an imported version to select its objects;
   click one that was not imported (the panel asks first when results are large) to import it — nothing is downloaded
   again. Right-click opens its folder. The files are under the Maya project's `data\lab2shot\<node name>\v001\`.
7. Select a Lab2Shot node in the scene and the panel switches to it; **Tasks in This Scene** on the home page lists
   every node of the scene too. **< All Tools** at the top goes back to the home page. After closing and reopening the
   scene, **… > Fetch Results** at the top right of the tool page fetches the last results again (the server keeps
   them for a few days).
8. Language: **… > Language** at the top right of the panel: follow Maya, 中文 or English. The panel and the menu
   change at once.

## Troubleshooting

- The plug-in's own log is `C:\Users\<your user name>\lab2shot\maya.log` (**… > Log Folder** in the panel, or
  **Log Folder** in the menu). An error in the plug-in is written to the log and said in the panel; it never affects
  Maya.
- No menu: check the log above; make sure `lab2shot.mod` and the `lab2shot` folder are in the same `modules` folder.
- Cannot connect to the server: check the address. When the server uses an HTTPS certificate it signed itself, the
  package already carries it (`lab2shot\scripts\lab2shot_ca.pem`) and the plug-in trusts it as well.
