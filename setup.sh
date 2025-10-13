#!/bin/bash

# DS Capstone Project Setup Script
# This script sets up the development environment for the multi-agent classification system

set -e

echo "🚀 Setting up DS Capstone Multi-Agent Classification System"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is required but not installed."
        exit 1
    fi
    
    # Check Docker (optional)
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. You'll need it for production deployment."
    fi
    
    # Check Docker Compose (optional)
    if ! command -v docker-compose &> /dev/null; then
        print_warning "Docker Compose is not installed. You'll need it for production deployment."
    fi
    
    print_success "System requirements check completed"
}

# Setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    cd backend
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Create necessary directories
    mkdir -p uploads temp logs
    
    print_success "Backend setup completed"
    cd ..
}

# Setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd frontend
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    print_success "Frontend setup completed"
    cd ..
}

# Setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            print_success "Environment file created from template"
            print_warning "Please edit .env file with your actual API keys and configuration"
        else
            print_error "env.example file not found"
            exit 1
        fi
    else
        print_warning "Environment file already exists, skipping creation"
    fi
}

# Setup database
setup_database() {
    print_status "Setting up database..."
    
    # Check if PostgreSQL is running
    if ! pg_isready -q; then
        print_warning "PostgreSQL is not running. Please start PostgreSQL before running the application."
        print_status "You can start PostgreSQL with: brew services start postgresql (on macOS)"
        return
    fi
    
    # Create database if it doesn't exist
    createdb ds_capstone 2>/dev/null || print_warning "Database 'ds_capstone' might already exist"
    
    print_success "Database setup completed"
}

# Setup Redis
setup_redis() {
    print_status "Setting up Redis..."
    
    # Check if Redis is running
    if ! redis-cli ping &> /dev/null; then
        print_warning "Redis is not running. Please start Redis before running the application."
        print_status "You can start Redis with: brew services start redis (on macOS)"
        return
    fi
    
    print_success "Redis setup completed"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    cd backend
    source venv/bin/activate
    
    # Initialize Alembic if not already done
    if [ ! -d "alembic" ]; then
        alembic init alembic
    fi
    
    # Run migrations
    alembic upgrade head
    
    print_success "Database migrations completed"
    cd ..
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
EOF
    chmod +x start_backend.sh
    
    # Frontend startup script
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
npm run dev
EOF
    chmod +x start_frontend.sh
    
    # Celery worker startup script
    cat > start_celery.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
celery -A app.celery_app worker --loglevel=info
EOF
    chmod +x start_celery.sh
    
    print_success "Startup scripts created"
}

# Main setup function
main() {
    echo
    print_status "Starting setup process..."
    echo
    
    check_requirements
    setup_environment
    setup_backend
    setup_frontend
    setup_database
    setup_redis
    run_migrations
    create_startup_scripts
    
    echo
    print_success "🎉 Setup completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Edit .env file with your API keys"
    echo "2. Start the backend: ./start_backend.sh"
    echo "3. Start the frontend: ./start_frontend.sh"
    echo "4. Start Celery worker: ./start_celery.sh"
    echo "5. Open http://localhost:3000 in your browser"
    echo
    echo "For production deployment with Docker:"
    echo "1. docker-compose up -d"
    echo
    print_status "Happy coding! 🚀"
}

# Run main function
main "$@"
