LABEL_NORMALIZATION_DICT = {
    "Presenza di vegetazione": "vegetazione",
    "Presenza di Vegetazione": "vegetazione",
    "Porzione di muratura mancante": "muratura_mancante",
    "Nessun difetto": "nessun_difetto",
    "Fratturazione o Fessurazione": "fratturazione_fessurazione",
    "Fratturazione o fessurazione": "fratturazione_fessurazione",
    "Fessurazione o fratturazione": "fratturazione_fessurazione"
}

LABEL_TO_ID = {
    "background": 0,
    "vegetazione": 1,
    "muratura_mancante": 2,
    "nessun_difetto": 3,
    "fratturazione_fessurazione": 4
}

ID_TO_LABEL = { value: key for key, value in LABEL_TO_ID.items() }

ID_TO_COLOR = {
    0: [0, 0, 0],           # background
    1: [0, 255, 0],         # vegetazione
    2: [255, 0, 0],         # muratura_mancante
    3: [255, 255, 0],       # nessun_difetto
    4: [0, 0, 255],         # fratturazione_fessurazione
}