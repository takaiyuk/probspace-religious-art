# Probspace - 宗教画テーマの分類

https://prob.space/competitions/religious_art

## Run

```
$ ./scripts/docker/run.sh
$ ./scripts/docker/exec.sh
root@xxxxx:/workspace# venv-acitvate
(venv) root@xxxxx:/workspace# ./scrirpts/run.sh expXXX
```

## Jupyter

```
$ ./scripts/docker/run.sh
$ ./scripts/docker/exec.sh
root@xxxxx:/workspace# venv-acitvate
(venv) root@xxxxx:/workspace# ./scripts/jupyter.sh
```

## Submission

### Late Sub

Name | CV | Public Score | Private Score | Base | Description
-- | -- | -- | -- | -- | --
submission_036.csv | 0.66055 | 0.625 | 0.630 | exp034 | stacking: exp008, exp010, exp012, exp015, exp018 (2nd stage: vote)

### Result

Name | CV | Public Score | Private Score | Base | Description
-- | -- | -- | -- | -- | --
submission_035.csv | ------- | ----- | - | - | stacking: exp018, exp015, exp012, exp010 (2nd stage)
submission_034.csv | 0.66055 | 0.625 | **0.633** | exp027 | stacking: exp018, exp015, exp012, exp010 (weighten with cv score)
submission_033.csv | 0.56061 | 0.578 | 0.607 | exp022 | Larger image size
submission_032.csv | 0.65749 | 0.625 | 0.637 | - | stacking: exp028, exp027, exp026 (average)
submission_031.csv | 0.65291 | 0.641 | 0.630 | - | stacking: exp028, exp027, exp026, exp025 (average)
submission_030.csv | 0.64220 | 0.641 | 0.624 | - | stacking: exp028, exp027, exp026, exp025, exp024 (average)
submission_029.csv | 0.63914 | 0.625 | 0.624 | - | stacking: exp028, exp027, exp026, exp025, exp024, exp023 (average)
submission_028.csv | 0.64832 | 0.609 | - | - | same as submission_021
submission_027.csv | 0.66055 | 0.656 | - | - | same as submission_020
submission_026.csv | 0.65291 | 0.641 | 0.630 | - | same as submission_019
submission_025.csv | 0.61927 | 0.656 | - | - | same as submission_016
submission_024.csv | 0.61927 | 0.609 | - | - | same as submission_013
submission_023.csv | 0.59786 | 0.625 | 0.612 | - | same as submission_011
submission_022.csv | ------- | ----- | - | exp018 | CutMix
submission_021.csv | 0.64832 | 0.609 | 0.630 | - | stacking: exp012, exp015, exp018 (average)
submission_020.csv | 0.66055 | 0.656 | **0.635** | - | stacking: exp010, exp012, exp015, exp018 (average)
submission_019.csv | 0.65291 | 0.656 | 0.630 | - | stacking: exp008, exp010, exp012, exp015, exp018 (average)
submission_018.csv | 0.64394 | 0.609 | ***0.640*** | exp017 | leak fix (avoid evaluating psuedo labeled dataset)
submission_017.csv | 0.72126 | 0.578 | 0.635 | exp015 | psuedo labeling
submission_016.csv | 0.61927 | 0.656 | 0.619 | - | stacking: exp008, exp010, exp012, exp015 (average)
submission_015.csv | 0.60550 | 0.609 | 0.630 | exp014 | resnext50 -> resnext101
submission_014.csv | 0.59939 | 0.562 | 0.612 | exp012 | larger image_size, more epochs
submission_013.csv | 0.61927 | 0.609 | 0.619 | - | stacking: exp008, exp010, exp012 (average)
submission_012.csv | 0.60550 | 0.609 | 0.594 | exp008 | resnext50, more augmentations
submission_011.csv | 0.59786 | 0.625 | 0.612 | - | stacking: exp008, exp010 (average)
submission_010.csv | 0.58716 | 0.578 | 0.582 | exp008 | resnest50
submission_009.csv | 0.50765 | - | - | exp008 | mobilenetv3
submission_008.csv | 0.57034 | 0.594 | 0.561 | exp006 | image サイズを大きく
submission_007.csv | 0.56422 | 0.484 | 0.497 | exp006 | efficient_b2 に変更
submission_006.csv | 0.55657 | 0.516 | 0.545 | exp005 | 損失関数に weights を追加（少数クラスほど weight を大きく）
submission_005.csv | 0.50917 | - | - | exp002 | model を resnet50 に変更
submission_004.csv | 0.48165 | 0.484 | 0.547 | exp002 | 重複画像のラベルデータを後処理で埋める
submission_003.csv | 0.46636 | - | - | exp002 | 輝度（Brightness）の統一
submission_002.csv | 0.48165 | 0.484 | 0.545 | - | -

### Probing

y | CV | train sample size | Public | public sample size |
-- | -- | -- | -- | --
0 | 0.092 | 60 | 0.078 | 5
1 | 0.064 | 42 | 0.078 | 5
2 | 0.202 | 132 | 0.156 | 10
3 | 0.064 | 42 | 0.109 | 7
4 | 0.064 | 42 | 0.078 | 5
5 | 0.092 | 60 | 0.047 | 3
6 | 0.073 | 48 | 0.078 | 5
7 | 0.046 | 30 | 0.062 | 4
8 | 0.046 | 30 | 0.031 | 2
9 | 0.101 | 66 | 0.078 | 5
10 | 0.046 | 30 | 0.047 | 3
11 | 0.064 | 42 | 0.094 | 6
12 | 0.046 | 30 | 0.062 | 4
