#!/bin/bash

# Create necessary directories
mkdir -p milvus/data milvus/conf milvus/logs
mkdir -p etcd/data
mkdir -p minio/data
echo "docker-compose up -d"
