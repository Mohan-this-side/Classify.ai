#!/bin/bash

# DS Capstone Multi-Agent System Startup Script
# This script starts the entire system using Docker Compose

set -e

echo "🚀 Starting DS Capstone Multi-Agent System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

# Check if .env file exists in root directory
if [ ! -f .env ]; then
    echo "⚠️  .env file not found in root directory. Creating from example..."
    if [ -f config/env.example ]; then
        cp config/env.example .env
        echo "📝 Please edit .env file with your API keys before continuing."
        echo "   Required: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
        read -p "Press Enter to continue after editing .env file..."
    else
        echo "❌ config/env.example not found. Please create .env file manually."
        exit 1
    fi
fi

# Navigate to docker directory
cd docker

# Copy .env from root to docker directory for docker-compose
if [ -f ../.env ]; then
    echo "📋 Copying .env file to docker directory..."
    cp ../.env .env
else
    echo "❌ .env file not found in root directory."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p ../backend/uploads
mkdir -p ../backend/temp
mkdir -p ../backend/results
mkdir -p ../backend/models
mkdir -p ../backend/notebooks
mkdir -p ../backend/plots

# Start services
echo "🐳 Starting Docker services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
fi

# Check Backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is ready"
else
    echo "❌ Backend API is not ready"
fi

# Check Frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is ready"
else
    echo "❌ Frontend is not ready"
fi

echo ""
echo "🎉 System startup complete!"
echo ""
echo "📊 Service URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   PostgreSQL: localhost:5432"
echo "   Redis: localhost:6379"
echo ""
echo "📝 To view logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 To stop the system:"
echo "   docker-compose down"
echo ""
echo "🔧 To restart a specific service:"
echo "   docker-compose restart <service-name>"
echo ""
echo "Happy coding! 🚀"