CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
TARGET = haldane
SOURCE = haldane.cpp

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

.PHONY: all clean install test

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

test: $(TARGET)
	@echo "Running test for spin-1/2 chain..."
	./$(TARGET) 0.5 8 30 0 4
	@echo "Running test for spin-1 chain..."
	./$(TARGET) 1 6 30 0 4

help:
	@echo "Available targets:"
	@echo "  all     - Build the haldane executable (default)"
	@echo "  clean   - Remove built files"
	@echo "  install - Install to /usr/local/bin"
	@echo "  test    - Run basic tests"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make                    # Build with auto-detected Eigen3"
	@echo "  make EIGEN_PATH=/path   # Build with custom Eigen3 path"
