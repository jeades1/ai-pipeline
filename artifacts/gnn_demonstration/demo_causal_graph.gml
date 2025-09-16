graph [
  node [
    id 0
    label "creatinine_mean"
  ]
  node [
    id 1
    label "creatinine_max"
  ]
  node [
    id 2
    label "creatinine_slope"
  ]
  node [
    id 3
    label "urea_mean"
  ]
  node [
    id 4
    label "urea_max"
  ]
  node [
    id 5
    label "sodium_mean"
  ]
  node [
    id 6
    label "potassium_mean"
  ]
  node [
    id 7
    label "chloride_mean"
  ]
  node [
    id 8
    label "hemoglobin_mean"
  ]
  node [
    id 9
    label "platelets_mean"
  ]
  node [
    id 10
    label "wbc_mean"
  ]
  node [
    id 11
    label "module_injury"
  ]
  node [
    id 12
    label "module_repair"
  ]
  node [
    id 13
    label "gene_HAVCR1"
  ]
  node [
    id 14
    label "gene_LCN2"
  ]
  edge [
    source 0
    target 1
    confidence 0.9
    weight 0.9
  ]
  edge [
    source 0
    target 3
    confidence 0.8
    weight 0.8
  ]
  edge [
    source 0
    target 11
    confidence 0.7
    weight 0.7
  ]
  edge [
    source 0
    target 8
    confidence -0.4
    weight -0.4
  ]
  edge [
    source 1
    target 4
    confidence 0.85
    weight 0.85
  ]
  edge [
    source 1
    target 2
    confidence 0.7
    weight 0.7
  ]
  edge [
    source 2
    target 13
    confidence 0.65
    weight 0.65
  ]
  edge [
    source 3
    target 11
    confidence 0.6
    weight 0.6
  ]
  edge [
    source 3
    target 8
    confidence -0.35
    weight -0.35
  ]
  edge [
    source 4
    target 14
    confidence 0.6
    weight 0.6
  ]
  edge [
    source 5
    target 7
    confidence 0.6
    weight 0.6
  ]
  edge [
    source 5
    target 6
    confidence 0.4
    weight 0.4
  ]
  edge [
    source 8
    target 9
    confidence 0.3
    weight 0.3
  ]
  edge [
    source 9
    target 10
    confidence 0.25
    weight 0.25
  ]
  edge [
    source 11
    target 13
    confidence 0.8
    weight 0.8
  ]
  edge [
    source 11
    target 14
    confidence 0.75
    weight 0.75
  ]
  edge [
    source 11
    target 12
    confidence 0.5
    weight 0.5
  ]
]
