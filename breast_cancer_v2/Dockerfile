# Reproducible base — pinned dated tag from quay.io/jupyter/scipy-notebook
FROM quay.io/jupyter/scipy-notebook:2026-04-27

# scipy-notebook already includes pandas, scikit-learn, seaborn, matplotlib, numpy.
# Only SHAP is missing for the breast cancer classification notebook.
RUN pip install --no-cache-dir shap

WORKDIR /home/jovyan/work
