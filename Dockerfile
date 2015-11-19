FROM b.gcr.io/tensorflow/tensorflow
RUN pip install matplotlib
WORKDIR /app
CMD ["python"]