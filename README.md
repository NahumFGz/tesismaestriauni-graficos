# tesismaestriauni-graficos

Repositorio con **cuadernos Jupyter** para generar las figuras de resultados experimentales de la tesis de maestría (clasificación, detección, OCR, RAG, text-to-SQL y evaluación de respuestas con LLM). Los valores numéricos están **incrustados en cada notebook**; al ejecutarlo se reproducen los gráficos y se pueden exportar a PNG en alta resolución.

## Dependencias

- Python 3
- Paquetes: `jupyter`, `matplotlib`, `numpy` (ver [`requirements.txt`](requirements.txt))

```bash
pip install -r requirements.txt
jupyter notebook
```

## Cuadernos y salidas

| Notebook                                               | Contenido                                                                                                                                                                                | Archivo PNG (por defecto)          |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| [`1_clasificador.ipynb`](1_clasificador.ipynb)         | Accuracy de validación con IC del 95 % y resultado en **test** para redes (DenseNet121, EfficientNet-B0, MobileNet v2, ResNet50, VGG16) ante distintos learning rates y configuraciones. | `1_clasificador.png`               |
| [`2_yolo_asistencia.ipynb`](2_yolo_asistencia.ipynb)   | **mAP@50–95** (IC 95 %) para variantes **YOLO11** (n/s/l/x) en detección relacionada con **asistencia**, según tamaño de imagen y batch.                                                 | `2_yolo_asistencia.png`            |
| [`3_yolo_votacion.ipynb`](3_yolo_votacion.ipynb)       | Igual que el anterior pero para el escenario de **votación**. Resalta el mejor modelo por mAP medio.                                                                                     | `3_yolo_votacion.png`              |
| [`4_ocr_cer.ipynb`](4_ocr_cer.ipynb)                   | **CER** por zona del acta (**Tesseract**, **EasyOCR**, **docTR**, **PaddleOCR**): Asunto, encabezados, presidente, agregado, etc.; menor es mejor.                                       | `4_ocr_cer.png`                    |
| [`5_ocr_wer.ipynb`](5_ocr_wer.ipynb)                   | **WER** con la misma estructura y motores OCR que el cuaderno de CER.                                                                                                                    | `5_ocr_wer.png`                    |
| [`6_rag_asistencia.ipynb`](6_rag_asistencia.ipynb)     | Recuperación **RAG**: **Qdrant** frente a **BM25** y **TF-IDF** (Precision@10, Recall@10, Top-1, MAP, MRR con IC 95 %) en contexto **asistencia**.                                       | `6_rag_asistencia.png`             |
| [`7_rag_votacion.ipynb`](7_rag_votacion.ipynb)         | Mismas métricas y métodos para **votación**.                                                                                                                                             | `7_rag_votacion.png`               |
| [`8_rag_low.ipynb`](8_rag_low.ipynb)                   | RAG en el escenario de **menor recurso / métricas más bajas** (nombre `rag_low`; mismos comparativos que 6–7).                                                                           | `8_rag_low.png`                    |
| [`9_txt_to_sql.ipynb`](9_txt_to_sql.ipynb)             | Comparación de enfoques (**Script General**, **Agente Simple**, **Nuestro Agente**) en puntuaciones **EX** y **FLEX**.                                                                   | `9_txt_to_sql.png`                 |
| [`10_llm.ipynb`](10_llm.ipynb)                         | Métricas de calidad conversacional (**Ragas** y métricas clásicas) por dominio: Asistencia, Votación, Contrataciones y agregado.                                                         | `conversational_metrics_final.png` |
| [`11_llm_all_metrics.ipynb`](11_llm_all_metrics.ipynb) | Mismos datos que `10_llm`, pero **partidos en dos figuras**: RAGAS (`11_llm_split_ragas.png`) y métricas clásicas (`11_llm_split_classic.png`).                                          | varios PNG                         |

## Notas

- Todas las figuras usan **intervalos de confianza al 95 %** en formato errorbar donde aplica.
- Para actualizar una figura hay que editar los arreglos `rows` del cuaderno correspondiente y volver a ejecutar las celdas (y `plt.savefig(...)` si se desea sobrescribir el PNG).
