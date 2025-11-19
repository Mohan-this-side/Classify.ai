# Deployment Guide

This guide provides comprehensive instructions for deploying the DS Capstone Multi-Agent Classification System.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Manual Deployment](#manual-deployment)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Database Setup](#database-setup)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Production Considerations](#production-considerations)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Docker**: Version 20.10 or higher
- **Docker Compose**: Version 2.0 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: Minimum 20GB free space
- **CPU**: 4+ cores recommended

### API Keys Required

- **Google Gemini API Key**: For AI model operations
- **OpenAI API Key**: For fallback AI operations (optional)
- **Anthropic API Key**: For Taskmaster operations (optional)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ds-capstone-project
```

### 2. Set Up Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys
nano .env
```

### 3. Start the System

```bash
# Make startup script executable
chmod +x start_system.sh

# Start the system
./start_system.sh
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Navigate to docker directory**:
   ```bash
   cd docker
   ```

2. **Set up environment**:
   ```bash
   # Copy environment template
   cp ../docker.env.example .env
   
   # Edit with your API keys
   nano .env
   ```

3. **Start services**:
   ```bash
   docker-compose up -d
   ```

4. **Check service status**:
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

### Service Architecture

The system consists of the following services:

- **postgres**: PostgreSQL database
- **redis**: Redis cache and message broker
- **backend**: FastAPI backend service
- **celery-worker**: Celery worker for background tasks
- **celery-beat**: Celery scheduler
- **frontend**: Next.js frontend application
- **ml-sandbox**: Secure ML code execution environment
- **nginx**: Reverse proxy (optional)

### Stopping the System

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: This will delete all data)
docker-compose down -v
```

## Manual Deployment

### Backend Setup

1. **Create virtual environment**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp ../env.example .env
   # Edit .env with your configuration
   ```

4. **Start the backend**:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Set up environment**:
   ```bash
   cp ../env.example .env.local
   # Edit .env.local with your configuration
   ```

3. **Start the frontend**:
   ```bash
   npm run dev
   ```

### Database Setup

1. **Install PostgreSQL**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   
   # macOS with Homebrew
   brew install postgresql
   
   # Windows
   # Download from https://www.postgresql.org/download/windows/
   ```

2. **Create database**:
   ```sql
   CREATE DATABASE ds_capstone;
   CREATE USER ds_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE ds_capstone TO ds_user;
   ```

3. **Install Redis**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # macOS with Homebrew
   brew install redis
   
   # Windows
   # Download from https://github.com/microsoftarchive/redis/releases
   ```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | - | Yes |
| `GOOGLE_API_KEY` | Google API key (fallback) | - | No |
| `OPENAI_API_KEY` | OpenAI API key (fallback) | - | No |
| `ANTHROPIC_API_KEY` | Anthropic API key (Taskmaster) | - | No |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:password@localhost/ds_capstone` | Yes |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` | Yes |
| `CELERY_BROKER_URL` | Celery broker URL | `redis://localhost:6379/0` | Yes |
| `SECRET_KEY` | Secret key for JWT tokens | - | Yes |
| `DEBUG` | Enable debug mode | `false` | No |
| `MAX_FILE_SIZE` | Maximum file upload size (bytes) | `104857600` (100MB) | No |

### Database Configuration

The system uses PostgreSQL as the primary database. Configure the connection string:

```bash
DATABASE_URL=postgresql://username:password@host:port/database_name
```

### Redis Configuration

Redis is used for caching and as a message broker for Celery:

```bash
REDIS_URL=redis://username:password@host:port/database_number
```

## Monitoring

### Health Checks

The system provides health check endpoints:

- **Backend Health**: `GET /health`
- **Database Health**: `GET /health/database`
- **Redis Health**: `GET /health/redis`

### Logging

Logs are available through Docker Compose:

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f celery-worker
```

### Metrics

The system includes basic metrics collection:

- **Prometheus Metrics**: Available at `/metrics` (if enabled)
- **Custom Metrics**: Workflow execution times, success rates, error counts

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Check what's using the port
lsof -i :8000
lsof -i :3000

# Kill the process
kill -9 <PID>
```

#### 2. Database Connection Failed

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -h localhost -U postgres -d ds_capstone
```

#### 3. Redis Connection Failed

```bash
# Check Redis status
sudo systemctl status redis

# Test connection
redis-cli ping
```

#### 4. API Key Issues

- Verify API keys are correctly set in `.env` file
- Check API key permissions and quotas
- Ensure API keys are not expired

#### 5. Memory Issues

```bash
# Check memory usage
docker stats

# Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# In .env file
DEBUG=true
LOG_LEVEL=DEBUG
```

### Reset System

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: Deletes all data)
docker-compose down -v

# Rebuild and start
docker-compose up --build -d
```

## Production Considerations

### Security

1. **Change default passwords**:
   ```bash
   # Update PostgreSQL password
   POSTGRES_PASSWORD=your_secure_password
   
   # Update secret key
   SECRET_KEY=your_very_secure_secret_key
   ```

2. **Use HTTPS**:
   - Configure SSL certificates
   - Update `NEXT_PUBLIC_API_URL` to use HTTPS
   - Update `NEXT_PUBLIC_WS_URL` to use WSS

3. **Network Security**:
   - Use firewall rules
   - Limit database access
   - Use VPN for remote access

### Performance

1. **Resource Allocation**:
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       limits:
         cpus: '2.0'
         memory: 4G
   ```

2. **Database Optimization**:
   - Configure PostgreSQL for production
   - Set up connection pooling
   - Enable query optimization

3. **Caching**:
   - Configure Redis for optimal performance
   - Use CDN for static assets
   - Implement application-level caching

### Backup

1. **Database Backup**:
   ```bash
   # Create backup
   pg_dump -h localhost -U postgres ds_capstone > backup.sql
   
   # Restore backup
   psql -h localhost -U postgres ds_capstone < backup.sql
   ```

2. **File Storage Backup**:
   - Backup uploaded files
   - Backup generated models and reports
   - Implement automated backup schedule

### Scaling

1. **Horizontal Scaling**:
   - Add more Celery workers
   - Use load balancer for multiple backend instances
   - Implement database read replicas

2. **Vertical Scaling**:
   - Increase memory and CPU allocation
   - Use faster storage (SSD)
   - Optimize database configuration

### Monitoring and Alerting

1. **Application Monitoring**:
   - Set up Prometheus and Grafana
   - Monitor API response times
   - Track error rates and success rates

2. **Infrastructure Monitoring**:
   - Monitor CPU, memory, and disk usage
   - Set up alerts for resource thresholds
   - Monitor database performance

3. **Log Aggregation**:
   - Use ELK stack (Elasticsearch, Logstash, Kibana)
   - Centralize log collection
   - Implement log analysis and alerting

## Support

For additional support:

1. **Documentation**: Check the main README.md
2. **Issues**: Create an issue in the repository
3. **Discussions**: Use GitHub Discussions for questions
4. **Email**: Contact the development team

## License

This project is licensed under the MIT License. See the LICENSE file for details.
