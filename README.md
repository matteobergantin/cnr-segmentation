### Segmentation Project

Ho riportato tutto il codice dal mio colab in python, cercando di strutturare tutto nel progetto.

Devo ancora aggiungere commenti e riscrivere parte del codice per renderla più leggibile.

Per il colab originale andate [qui](https://colab.research.google.com/drive/1Uc4_8aXpzdLAJMu35bIXybzfXlhyayIb?usp=sharing)

### Come utilizzare il modello
Scaricate il modello fine-tunato da me [qui](https://drive.google.com/file/d/18p9lFG1hZiIG1YF2wyPGaE4fSJbVfMHO/view?usp=sharing), e mettetelo nella directory `output/trained_models` con il nome `weights.pth`.

Installate le librerie necessarie con:
```bash
pip install -r requirements.txt
```

Create una cartella `input_data` e metteteci dentro una qualsiasi immagine vogliate valutare.

Lanciate `main.py` e lui prenderà le immagini inserite in `input_data` e ci lancerà sopra il modello, i risultati li troverete in `output/generated_images`.