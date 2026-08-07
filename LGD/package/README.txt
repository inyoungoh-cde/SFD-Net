LGD - Geometric Feature Extractor (portable)
================================================

WHAT IT DOES
  Reads point-cloud .txt files (whitespace-separated, one point per line),
  computes 3-channel geometric features per point, and writes results as:

      x y z  f0 f1 f2  L

  where L is the label taken from the LAST column of the input file.
  (If the input has only 3 columns (x y z), no label is written.)

HOW TO RUN
  Double-click LGD.exe (or run it without arguments) to enter
  interactive mode: you will be asked for the input folder, output folder
  and k one by one. Press Enter to accept the default shown in [brackets].

  Or pass everything on the command line:

  LGD.exe [input_dir] [output_dir] [k]

    input_dir   folder containing the input .txt files   (default: data)
    output_dir  folder where results are written          (default: output_YYMMDD)
    k           k-NN neighborhood size                    (default: 40)

  * Relative paths are resolved against the folder containing LGD.exe.
  * Argument order is flexible: a purely numeric argument is taken as k;
    the first non-numeric argument is the input folder, the second is the
    output folder.
  * Output files keep the same file names as the input files.

EXAMPLES
  LGD.exe
      -> reads .\data, writes .\output_<today>, k = 40

  LGD.exe 80
      -> reads .\data, writes .\output_<today>, k = 80

  LGD.exe mydata results 20
      -> reads .\mydata, writes .\results, k = 20

  LGD.exe D:\pointclouds D:\features 40
      -> absolute paths are used as-is

  LGD.exe -h
      -> prints usage

REQUIREMENTS
  - 64-bit Windows 10 or later.
  - The required Visual C++ runtime DLLs (msvcp140.dll, vcruntime140.dll,
    vcomp140.dll) are included in this folder; keep them next to LGD.exe.

INPUT FORMAT
  Each .txt file: one point per line, at least 3 numeric columns (x y z).
  Extra columns are allowed; the last column is treated as the label.
  All lines in a file must have the same number of columns
  (mismatching lines are skipped with a warning).

NOTE
  Each point cloud should contain at least (2 * (k + 1)) points, since the
  method uses a multi-scale neighborhood up to twice the requested k.
