# TOPOLOGICAL CROSSOVER DETECTION IN SPATIAL GROWTH SYSTEMS
This repository contains the simulation source code, data extraction pipeline, and numerical validation scripts for the research paper:

Paper Title: A Loop-Formation-Rate Criterion for Detecting Topological Crossover in Growth-Limited Systems

Author: Nazwa Shabrina Zain (Independent Researcher)

Paper Link: [A Loop-Formation-Rate Criterion for Detecting Topological Crossover in Growth-Limited Systems](https://www.academia.edu/170698334/A_Loop_Formation_Rate_Criterion_for_Detecting_Topological_Crossover_in_Growth_Limited_Systems)

## METHODOLOGICAL HIERARCHY & SCOPE
Important Note on System Modeling:

"Sistem pertumbuhan dimodelkan melalui suatu operator evolusi M, yang digunakan sebagai mekanisme untuk menghasilkan konfigurasi spasial pada setiap langkah simulasi. Dalam penelitian ini, operator tersebut bukan merupakan objek utama analisis, melainkan sarana untuk menghasilkan data yang kemudian dianalisis menggunakan indikator topologi dan geometri."

This study demonstrates that combining topological indicators (such as the loop-formation rate $$g(V) = d\beta_1/dV$$) and geometric scaling metrics (E(V) = d ln A / d ln V) yields a significantly more robust crossover criterion compared to traditional curve-shape fitting or arbitrary scalar thresholds.

## REPOSITORY STRUCTURE
```
topological_crossover_detection
│  ansatz.png
│  ansatz.py
│  ansatz_benar.png
│  ansatz_benar.py
│  membuktikan_transisi_rezim_mekanis.png
│  membuktikan_transisi_rezim_mekanis.py
│  README.md
│  uji_formulasi_laju_maksimum_v.png
│  uji_formulasi_laju_maksimum_v.py
│
└─metrologi
```

## EXECUTION & VERIFICATION PIPELINE

### Step 1: Robustness & Uniqueness Verification
To test why V_dagger = arg max g(V) outperforms traditional second-derivative inflection criteria on complex/non-sigmoidal trajectories, run:
```
python uji_formulasi_laju_maksimum_v.py
```

### Step 2: Geometric Scale Transition Verification
To observe the post-crossover mechanical boundary saturation (E(V) -> 1.0 as V >> V_dagger), run:
```
python membuktikan_transisi_rezim_mekanis.py
```

## CITATION
If you use this code or numerical methodology in your research, please cite the paper as follows:
```
@article{zain2026loop,
  title={A Loop-Formation-Rate Criterion for Detecting Topological Crossover in Growth-Limited Systems},
  author={Zain, Nazwa Shabrina},
  journal={Independent Research},
  year={2026}
}
```

## LICENSE
Distributed under the MIT License. See LICENSE for more information.