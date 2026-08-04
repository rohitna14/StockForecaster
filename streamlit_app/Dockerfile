# Thin client image. Deliberately tiny -- it installs streamlit, httpx and
# plotly and nothing else. If this Dockerfile ever needs scikit-learn, the
# client/server separation has been broken.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN useradd --create-home --uid 1000 app

COPY streamlit_app/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=app:app streamlit_app/ .

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8501/_stcore/health').status_code==200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
