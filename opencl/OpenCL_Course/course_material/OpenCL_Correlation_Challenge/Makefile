
# Include general environment variables
include ../env

# List of applications to target
TARGETS=xcorr_answers.exe xcorr.exe xcorr_testbench.exe

all: $(TARGETS)

# General compilation step
%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $(BASE_INC_FLAGS) $< -o $@ $(BASE_LIB_FLAGS)

# Clean step
clean:
	rm -rf *.exe


.EXPORT_ALL_VARIABLES:
