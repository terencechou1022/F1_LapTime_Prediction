# Engineering Retrospective

Lessons that shaped this codebase — the decisions that survived contact with real data, and the mistakes that taught the most. Companion to [methodology.md](methodology.md), which covers *what* was built; this covers *why it ended up this way*.

## 1. Design the experiment so the result is attributable

The headline finding — wind generalizes cross-domain, temperature fails catastrophically — is only interesting if the difference can't be blamed on modeling choices. That constraint drove the architecture: both studies share the same 9-feature baseline, the same splits, the same grids, the same headline model, enforced by a template-method base class rather than by convention. When the two studies then behave differently, the mechanism and its training distribution are the only remaining explanations.

The discipline had teeth. An early feature, `TyreLifeNorm × TrackTemp`, improved fit — and was removed anyway, because it back-doored temperature into the wind model and quietly broke the "this model never sees the other mechanism" claim. Tree ensembles learn interactions from main effects; the isolation principle mattered more than a marginal metric gain.

## 2. Make the model refuse

The strategy layer (`f1lab/strategy.py`) withholds its correction term whenever the queried condition falls outside the feature range the model was trained on. This is the project's central design position: a regression will emit a number for any input, so *knowing when not to predict* has to be built, not hoped for.

Two details matter. First, the guard is precise about what it refuses — the pit-stop arithmetic (`net`, `gap`, the uncorrected decision) is always computed because it isn't a model output; only the model-dependent correction is withheld. Second, the refusal was validated empirically before being trusted: force-evaluating the temperature model on the out-of-support race yields R² = −6.08, an order of magnitude worse than guessing the mean. The guard doesn't just seem prudent; the evaluation study proves the number it suppresses would have been wrong.

## 3. Let the data kill your story

An early draft of the undercut scenario used a hand-picked wind value — chosen, uncomfortably, because it made the decision flip. That version was thrown away and the scenario was rebuilt to read its environmental inputs from the real 2025 race files: the Saudi headwind is the measured race maximum, the Las Vegas track temperature is the measured race mean, and the unmeasurable strategy-desk parameters (pit loss, gap, lap times) are explicitly labeled illustrative.

The rebuilt story turned out better than the invented one — the decision still flips, but now on a defensible number — and the episode became a working rule: if a value can be measured from data, measure it; if it can't, label it; and let the narrative follow the data rather than the other way around.

## 4. Grids that can answer questions

Hyperparameter grids here follow two rules: every library default is reachable (so "the default won" is a finding, not a search gap), and every axis carries margin beyond the plausible optimum (so a boundary pick is a diagnostic, not a dead end — if `n_estimators=2000` ever wins, `cv_results_` quantifies the gap to 1500 and answers whether the bound was binding). Both winning XGBoost configurations chose `learning_rate=0.05` and `max_depth=3` with room to spare on every axis — evidence the search was wide enough, documented in the grid's inline comments rather than in a notebook someone would lose.

## 5. Leakage is a design question, not a bug class

Three choices in this codebase exist purely to keep future information out of the model: `train_test_split(shuffle=False)` with `TimeSeriesSplit` so no fold trains on its own future; stint-position features computed *before* invalid-lap filtering so a lap's recorded stint position never depends on which other laps got dropped; and a rejected feature idea — sector times — because sector times sum to lap time and would have leaked the target into the inputs, however good they looked in a related classification paper. Each is one line of code; each was a deliberate decision recorded next to the line.

## 6. Tests encode the contracts you discovered, not the ones you wished for

Writing the test suite after the pipeline was mature surfaced behaviors nobody had written down: the target re-bases on the *surviving* best lap when the true stint-fastest was filtered out; the "drop the lap after an invalid lap" rule deliberately carries across a stint boundary; rows with non-positive normalized tyre life are silently discarded. The tests assert what the code actually does, with fixtures engineered to hit each edge — so the next refactor breaks loudly instead of silently changing the target definition. Sixteen deterministic tests on synthetic data now gate every push in CI, while the real 75–80 hour training run stays out of it; a `--quick` mode (tiny grids, mutually exclusive with `--save` so it can never overwrite the real models) smoke-tests the genuine pipeline in about fifteen seconds.

## 7. Numbers in prose rot

Every metric in the documentation traces to a generated artifact (`summary/metrics.csv`, `summary/best_params.csv`, or the deterministic console output of `scripts/mechanism.py`), and claims get re-verified against those artifacts whenever the docs change. A review pass caught a summary sentence asserting the three models' hold-out scores sat "within 0.07 of each other" when the actual spread was 0.136 — the defensible claim (the main model sits within 0.07 *of the best*) said something subtly different. The rule that followed: documentation is part of the change, updated in the same commit, and any numeric or directional claim gets recomputed from source before it ships.

## 8. Reproducibility is a feature, with a price tag on it

Determinism here is layered: seeds everywhere (`random_state=42`), an exact `==` dependency lock for reproducing the reported numbers alongside permissive floors in `pyproject.toml` for library users, one command (`python run.py`) that regenerates every table and figure, and an honest cost label on the full run (~75–80 hours) with a fifteen-second smoke path for everyone who just wants proof the pipeline works. Reproducibility claims that omit the price tag are marketing; the point of publishing the lock, the seeds, and the runtime table together is that someone could actually pay it.
