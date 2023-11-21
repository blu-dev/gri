GRI (game-ready-input) is a cross-platform input crate with the aim of making it simple and easy
to integrate into video games written in rust.

If you want to support this project, go wishlist Sentinels Inc. on Steam, which is the game this
library was created for.

# Cross-Platform Support Table
Legend:
- ✅ - Implemented
- ❌ - Not implemented yet

| Platform | GCC Support | Xbox Support | PS Support | FF Support |
|-|-|-|-|-|
| Linux | ✅ | | | |
| Windows | ✅ | | | |
| macOS | ✅ | | | |

## A Note on GameCube Controller Support
GCC support is feature gated on all platforms with the `gamecube` feature, as it uses rusb/libusb,
which means that you must license your code and game accordingly.

In the future, the ideal would be to change this to using native APIs (such as WinUSB or using udev)
to avoid these licensing constraints, however that is currently not a priority.

Instructions for all platforms on how to enable GameCube Controller support for libusb can be found
on the [Dolphin Emulator wiki](https://dolphin-emu.org/docs/guides/how-use-official-gc-controller-adapter-wii-u/).