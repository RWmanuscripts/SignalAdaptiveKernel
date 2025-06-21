# SignalAdaptiveKernelManuscript
To reproduce the figures and tables in our manuscript, run
```
bash rainfall.sh
bash kodak.sh
bash axis_plot sh
julia setup_manuscript.jl
```

The results should be stored in `tables`, `results`, `figs`. The table values should be printed to scrreen after `bash kodak.sh`.

To log the outputs of a bash script to `log.txt`, use 
```
bash script_name.sh |tee > log.txt
```

# System requirements
This repository was tested under Fedora Workstation 40, a 64-bit Linux operating system, with Julia `v1.10.4`. The file paths involving slash were done using `/`, which is designed for Linux. You might need to change all instances of `/` to `\` for file paths when you're on a Windows machine.

Alternatively, use Windows subsystem for Linux or virtual machines to run Linux under non-Linux operating systems.