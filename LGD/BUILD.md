# Building LGD

The same sources build on Windows and Linux. There are two build systems and
they compile identical code — pick whichever fits your platform.

| | Windows (Visual Studio) | Windows / Linux / macOS (CMake) |
|---|---|---|
| Project file | `LGD.sln`, `LGD.vcxproj` | `CMakeLists.txt` |
| Output | `x64\Release\LGD.exe` | `build/LGD` (`build\Release\LGD.exe` on MSVC) |
| Toolset | VS 2017 (v141), C++14 | any C++17 compiler |

## Requirements

* A C++17 compiler (or Visual Studio 2017, which uses the pre-standard
  `std::experimental::filesystem` — handled automatically in `platform.h`).
* OpenMP. Bundled with MSVC; on Ubuntu it comes with GCC.
* Nothing else: Eigen and nanoflann are header-only and already included in
  `third_party_includes/`.

## Windows — Visual Studio 2017

Open `LGD.sln` and press Ctrl+F5. The solution ships a single `Release|x64`
configuration, so it always builds and runs the optimized executable
(`x64\Release\LGD.exe`). Nothing else is produced.

## Making the redistributable Windows package (optional)

Only needed when publishing a ready-to-run build, e.g. for a GitHub release.
Double-click **`make_package.bat`** in the project root, or run:

```
msbuild LGD.sln /p:Configuration=Release /p:Platform=x64 /p:MakeDeployPackage=true
```

That produces:

```
deploy\LGD_portable\        LGD.exe + VC++ runtime DLLs + README + empty data\
deploy\LGD_portable.zip     the same folder zipped (make_package.bat only)
```

The packaging step is off by default (`MakeDeployPackage=false`), so ordinary
builds — and every CMake build, on any platform — never touch `deploy\`.
Everything in it is generated from `package\` plus the build output, so the
folder can be deleted at any time and rebuilt with the script.

## Linux / Ubuntu — CMake

```bash
sudo apt install build-essential cmake      # once
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/LGD data output 40
```

`cmake -S . -B build` needs CMake 3.13+. On older CMake use the classic form:

```bash
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
```

GCC 8 or newer is required for `std::filesystem` (Ubuntu 20.04 and later are
fine). GCC 7 falls back to `<experimental/filesystem>` automatically.

## Windows — CMake

```powershell
cmake -S . -B build -G "Visual Studio 15 2017" -A x64
cmake --build build --config Release
```

Note that this generator writes its own `LGD.sln` **inside** `build\` — open
the one in the project root, not that one. The CMake build does not update
`deploy\LGD_portable\`; that step belongs to the Visual Studio project.

## Running

```
LGD [input_dir] [output_dir] [k]
```

Run without arguments for an interactive prompt (Enter accepts each default).
Relative paths resolve against the project root, which the executable locates
by walking up from its own location until it finds `CMakeLists.txt` or
`third_party_includes/`. A deployed executable finds neither and uses its own
folder, which is why the portable package works anywhere.

## Portability notes

Everything OS-specific lives in `platform.h`:

* `std::filesystem` vs `std::experimental::filesystem` selection,
* executable path (`GetModuleFileNameA` vs `/proc/self/exe`),
* thread-safe local time (`localtime_s` vs `localtime_r`).

`algorithms.h` (the method itself), `runner.*` and `FileData.h` contain no
platform-specific code.

## Generated files (safe to delete)

`x64/`, `build/`, `.vs/`, `*.VC.db` are build artifacts and caches.
