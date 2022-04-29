docker build -t generate-landscape-service .
docker run --rm --gpus 1 -p 5001:5001 generate-landscape-service