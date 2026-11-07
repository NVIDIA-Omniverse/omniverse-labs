# Install the Blender add-on

The package includes the add-on and the software that runs rendering and physics
outside Blender. Install them together using the package's installer.

## Before you begin

- Install **Blender 5.2.1 LTS** separately.
- Download the complete **Windows x64** or **Linux x64** ZIP from the [Blender 0.2.0 release](https://github.com/NVIDIA-Omniverse/omniverse-labs/releases/tag/ovrtx-blender-example-0.2).
- Check the [GPU, driver and platform requirements](support-platforms.md).
- Save your work and close Blender. Allow space for the ZIP, extracted files and installation.

You do not need a separate Python installation or SDK download.

## 1. Extract the platform package

Extract the release ZIP into a folder. Keep its files together: the installer needs
both the add-on ZIP and the matching runtime ZIP. Do not select the outer release
ZIP in Blender's **Install from Disk** dialog.

If the release includes `SHA256SUMS`, use it to check that your download is intact.

## 2. Run the installer

Open a terminal in the extracted folder. Replace the example path with the location
of your Blender executable. Keep the quotation marks if the path contains spaces.

**Windows — Command Prompt:**

```bat
install.bat "C:\path\to\Blender\blender.exe"
```

**Linux:**

```sh
sh install.sh "/path/to/blender"
```

On Linux, use the `blender` file inside your Blender installation folder and run
from a desktop session with access to the NVIDIA GPU.

The installer enables the add-on and installs its matching runtime. Start Blender
normally once installation finishes.

## 3. Wait for renderer warm-up

The installer prepares the renderer by rendering a small scene. This can take
several minutes on the first run. Progress appears every 30 seconds.

If it exits with **code 3**, installation succeeded but warm-up failed. Read the
error and log locations printed in the terminal, then check your GPU, driver and
display setup. Keep the complete package together when retrying.

## 4. Verify in Blender

1. Open **Edit → Preferences → Add-ons** and confirm **`ovrtx` Blender Example** is enabled.
2. Open a scene with an object, a light and a camera. Blender's default scene is a simple starting point.
3. In **Render Properties**, choose **`ovrtx` Example** as the render engine.
4. Switch the 3D Viewport to **Rendered** shading and wait for an image.
5. If no image appears, read **Render Properties → `ovrtx` Example → `ovrtx` Session**.
6. Press **F12** to render the active camera.

Continue with [Render your first scene](render-guide.md) to adjust quality and save an image.

## Use a separate settings folder

Use a separate folder if you want to try the add-on without changing your usual
Blender setup. Pass that folder to both the installer and launcher.

**Windows:**

```bat
install.bat "C:\path\to\Blender\blender.exe" "C:\path\to\ovrtx-settings"
launch.bat "C:\path\to\Blender\blender.exe" "C:\path\to\ovrtx-settings"
```

**Linux:**

```sh
sh install.sh "/path/to/blender" "/path/to/ovrtx-settings"
sh launch.sh "/path/to/blender" "/path/to/ovrtx-settings"
```

Use the launcher each time you want this setup. Starting Blender normally uses
your usual settings instead.

## Upgrade or remove

To upgrade an existing 0.2 installation, close Blender and run the new package's
installer with the same Blender path and settings folder. Replace the complete
package rather than individual libraries. For an earlier integration, we recommend
using a [separate Blender settings folder](#use-a-separate-settings-folder).

To uninstall, choose **Remove Runtime** in the add-on preferences, then remove
the add-on through Blender's extension controls. This does not delete your scenes.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| Render engine is missing | Check that you opened the Blender version and settings folder used by the installer. Confirm the add-on is enabled. |
| Package or version error | Reinstall the complete matching release package. |
| Warm-up exits with code 3 | Read the error and log locations printed in the terminal. Check the GPU and display requirements. |
| Blank viewport | Select Rendered shading. Check that objects and lights are visible, then read **ovrtx Session**. |
| Rendering stops updating | Try **Reconnect ovrtx Session**, then **Restart ovrtx Worker** in Render Properties. |
| Installer reports a lock or incomplete installation | Close Blender and follow the recovery instructions in the installer message. Keep its log files. |
