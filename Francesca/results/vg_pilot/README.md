# VG Pilot Run

This folder contains the pilot run for the VG artefact scoring workflow.

Pilot artefacts:

- `S05`
- `M07`
- `L03`

Pilot scoring settings:

- 5 runs per artefact and scoring condition
- `domain_specific` + `neutral`
- `do_you_like` + `neutral`
- `domain_specific` + `anti_sycophantic`

The `neutrality/` subfolder contains the pilot neutrality batch input, downloaded
batch output, and parsed neutrality summary.

The `scoring/` subfolder contains the pilot scoring batch inputs, downloaded
batch outputs, parsed score rows, score summaries, baseline shifts, and plots.

The main `Francesca/results/vg_neutrality/` and
`Francesca/results/vg_scoring/` folders are now reserved for the full run.
