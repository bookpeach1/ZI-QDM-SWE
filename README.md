# ZI-QDM correction for daily snow water equivalent

This archive provides the Python code used to apply zero-inflated quantile
mapping (ZI-QM) to historical daily snow water equivalent (SWE) and
zero-inflated quantile delta mapping (ZI-QDM) to future daily SWE.

## Method retained from the research code

- Reference period: 1 January 1981 to 31 December 2020.
- Historical model correction: ZI-QM.
- Future model correction: ZI-QDM applied to 2021–2100 as one complete period.
- Occurrence threshold: `tau > 0`.
- Positive SWE correction: empirical quantile mapping in the `log1p` domain.
- Seasonal sampling: circular day-of-year window of ±15 days.
- Empirical probabilities clipped to 0.01–0.99.
- Grid cells are processed only when the first day is finite in both the
  reference and model files, as in the supplied research code.
- The occurrence threshold is trained once from the historical segment of the
  first scenario file and then applied to all scenario files.

The public version changes only file-path handling: local absolute paths have
been moved to a YAML configuration file. The correction equations, periods,
thresholds and processing sequence have not been changed.

## Files

- `zi_qdm_swe.py`: complete executable program.
- `config.example.yml`: path and execution configuration template.
- `environment.yml`: Conda environment.
- `LICENSE`: MIT software licence.
- `CITATION.cff`: citation metadata template.

## Installation

```bash
conda env create -f environment.yml
conda activate zi-qdm-swe
```

## Configuration

Copy `config.example.yml` to `config.yml`, then replace the example paths with
the paths to the ERA5-Land and model NetCDF files. The three scenario files
must use the same grid, time organization and historical model simulation.

The expected dimensions and coordinate names are `time`, `lat` and `lon`.
The default SWE variable names are `sd` for ERA5-Land and `snw` for the model.

## Run

```bash
python zi_qdm_swe.py --config config.yml
```

Each corrected file is written next to its input model file with the suffix
`_corrected.nc`. The intermediate occurrence-threshold array is written to the
location specified by `q_occ_tmp`.

## Input data

Raw ERA5-Land and CMIP6 data are not redistributed with this code. They should
be obtained from their original repositories and prepared on the same 0.25°
grid before running this program.

## Citation

After depositing these files in Zenodo, add the assigned DOI to
`CITATION.cff` and cite the software record together with the associated paper.

