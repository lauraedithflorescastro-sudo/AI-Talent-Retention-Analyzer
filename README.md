# AI Talent Retention Analyzer

Aplicación académica de People Analytics para **EV-Tech Mobility & Energy**, una empresa ficticia del sector de movilidad eléctrica, energía y manufactura avanzada.

## Qué hace

- Analiza rotación histórica.
- Entrena un modelo de Machine Learning para generar un **score relativo de riesgo**.
- Agrupa empleados actuales por segmentos de prioridad.
- Analiza entrevistas de salida y respuestas abiertas con TF-IDF.
- Cruza señales de salida con variables actuales para proponer áreas de investigación/intervención.
- Incluye un simulador *what-if* para explorar cómo cambia el score del modelo.

## Archivos

- `app.py`: aplicación principal.
- `requirements.txt`: dependencias.
- `AI_Talent_Retention_Synthetic_Data.xlsx`: dataset sintético.
- `Prompt_AI_Talent_Retention_Analyzer.txt`: prompt de diseño del proyecto.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Sube estos archivos a un repositorio de GitHub.
2. Crea una app nueva en Streamlit Community Cloud.
3. Selecciona la rama `main`.
4. Main file path: `app.py`.
5. Haz clic en **Deploy**.

No requiere API de OpenAI, Anthropic ni otro servicio de pago.

## Nota ética

Todos los datos son sintéticos. El modelo es demostrativo y no debe utilizarse para decisiones individuales de empleo. Los scores reflejan asociaciones del dataset y no prueban causalidad.