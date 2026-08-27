This folder includes the source code of the Oculus Quest APK delivered in the repository. Use it if you intend to customize the app. Before following this instruction, please read the [README.md](../README.md) in the repository root folder.

## Development

### Preparation

- (Mac only) Install Xcode
- Install Android development studio. On Ubuntu you can run `sudo snap install android-studio --classic`, otherwise follow: <https://developer.android.com/studio>
- Select *File -> Open...*, find the local path of the repo and open: app/build.gradle
- (Windows only) Install drivers https://developer.oculus.com/downloads/package/oculus-adb-drivers/

### Development

- The source code resides in the 'src/' directory.
- When ready, build the APK with menu Build -> Build App Bundle(s) / APK(s) -> Build APK(s). Locate file and replace the APK in the oculus_reader's APK directory.
- Reinstall APK with `python oculus_reader/install.py --reinstall`

## Run the code

Follow the instruction in [oculus_reader/README.md](../oculus_reader/README.md) to install the required packages, configure path to new the APK (as described in the same README file), and then finish up with the [README.md](../README.md) in the root folder to run the code.
