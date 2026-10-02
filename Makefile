CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
DEBUG_FLAGS = -std=c++17 -g -O0 -Wall -Wextra -DDEBUG
TARGET = haldane
SOURCE = haldane.cpp
VERSION = 1.0.0

# Try to find Eigen3 in common locations
EIGEN_PATHS = /usr/include/eigen3 /usr/local/include/eigen3 /opt/homebrew/include/eigen3 /opt/local/include/eigen3
EIGEN_PATH = $(firstword $(wildcard $(EIGEN_PATHS)))

ifneq ($(EIGEN_PATH),)
    CXXFLAGS += -I$(EIGEN_PATH)
else
    $(warning Eigen3 not found in standard locations. Please install Eigen3 or set EIGEN_PATH manually)
    $(warning Example: make EIGEN_PATH=/path/to/eigen3)
endif

# Allow user to override Eigen path
ifdef EIGEN_PATH_OVERRIDE
    CXXFLAGS += -I$(EIGEN_PATH_OVERRIDE)
endif

.PHONY: syk test-syk all debug clean install test help version

all: $(TARGET) syk

syk: syk.cpp
	$(CXX) -std=c++17 -O3 -Wall -Wextra syk.cpp -o syk

test-syk: syk
	./syk 12 20 1
	./syk 16 5 1

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

debug: $(SOURCE)
	$(CXX) $(DEBUG_FLAGS) $(SOURCE) -o $(TARGET)_debug

clean:
	rm -f $(TARGET) $(TARGET)_debug syk

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

test: $(TARGET)
	@echo "Running comprehensive tests..."
	@echo "Test 1: Spin-1/2 chain (OBC)..."
	./$(TARGET) 0.5 8 30 0 4
	@echo "Test 2: Spin-1 chain (OBC)..."
	./$(TARGET) 1 6 30 0 4
	@echo "Test 3: Spin-1/2 chain (PBC)..."
	./$(TARGET) 0.5 6 30 1 4
	@echo "Test 4: Spin-1 chain (PBC)..."
	./$(TARGET) 1 4 30 1 4
	@echo "All tests completed successfully!"

help:
	@echo "Haldane Chain Simulator - Makefile Help"
	@echo "========================================"
	@echo "Available targets:"
	@echo "  all     - Build optimized executable (default)"
	@echo "  debug   - Build debug version with symbols"
	@echo "  clean   - Remove all built files"
	@echo "  install - Install to /usr/local/bin"
	@echo "  test    - Run comprehensive test suite"
	@echo "  version - Show version information"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build with auto-detected Eigen3"
	@echo "  make debug              # Build debug version"
	@echo "  make EIGEN_PATH=/path   # Build with custom Eigen3 path"
	@echo "  make test               # Run all tests"

version:
	@echo "Haldane Chain Simulator v$(VERSION)"
	@echo "Build date: $(shell date)"
	@echo "Compiler: $(CXX) $(shell $(CXX) --version | head -n1)"
