Build:
docker build -t flask-web-service:latest .

Run:
docker run -d -p 5000:5000 flask-web-service

Access Service:
curl 'http://localhost:5000/'

View:
docker ps -a

Stop:
docker stop <CONTAINER ID>
