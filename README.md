# UR10 Sensitivity Analysis Error Simulation

This project implements a modular UR10 multi-source error simulation stack:

- `config/nominal_config.py`: base calibration, target-ball offset, nominal UR10 MD-H parameters, and per-scalar `is_optimizable` masks for later identification.
- `config/error_config.py`: base-calibration, target-ball, geometric, joint-flexibility, and measurement-noise distributions used to generate simulated truth models.
- `models/RobotModel.py`: config loading, Monte Carlo error injection, and Modified DH forward kinematics.
- `simulation/MeasurementSim.py`: target-ball measurement generation with laser tracker noise `sigma = A + B * d`.
- `calibration/Sim2Real_Calibrator.py`: sim-to-real RMSE alignment and calibrated config export.

Run the demo:

```bash
python main.py
```

Install dependencies:

```bash
pip install -r requirements.txt
```
