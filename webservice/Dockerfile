FROM ubuntu:latest
MAINTAINER William Milkovic "william.milkovic@student.fairfield.edu"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
COPY . /app
WORKDIR /app
RUN pip install -r reqs.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
