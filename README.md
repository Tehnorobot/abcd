# Run

```bash
python run_inference.py
```

# Важное
В `weights` нужно будет добавить вес модели ([скачать тут](https://drive.google.com/drive/folders/1zMowuO5pJApjnrcT5XkYHPAY4LYVtJvs?usp=sharing))

# Полезное
На выходе будет xlsx-файл (xlsx_bytes) в формате base64. Читать его так:

```
import io
import pandas as pd

xlsx_bytes = main()
df = pd.read_excel(io.BytesIO(xlsx_bytes))
df.head()
```

# Необходимое

В main нужно указать путь к zip-файлу, для которого будет происходит предсказание. ([Скачать тут](https://drive.google.com/drive/folders/1zMowuO5pJApjnrcT5XkYHPAY4LYVtJvs?usp=sharing))
