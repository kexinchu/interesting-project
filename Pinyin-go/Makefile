# init project path
HOMEDIR := $(shell pwd)
OUTDIR  := $(HOMEDIR)/output

# init command params
GO_1_16_BIN := "/home/work/.deck/1.0/go/1.16/bin"
GO      := $(GO_1_16_BIN)/go
GOROOT  := $(GO_1_16_HOME)
GOPATH  := $(shell $(GO) env GOPATH)
GOMOD   := $(GO) mod
GOBUILD := $(GO) build
GOTEST  := $(GO) test -gcflags="-N -l"
GOPKGS  := $$($(GO) list ./...| grep -vE "vendor")

# test cover files
COVPROF := $(HOMEDIR)/covprof.out  # coverage profile
COVFUNC := $(HOMEDIR)/covfunc.txt  # coverage profile information for each function
COVHTML := $(HOMEDIR)/covhtml.html # HTML representation of coverage profile

# make, make all
all: prepare compile package

# set proxy env
set-env:
	$(GO) env -w GO111MODULE=on
	$(GO) env -w GONOSUMDB=\*

#make prepare, download dependencies
prepare: gomod

gomod: set-env
	$(GOMOD) download

#make compile
compile: build

build:
	$(GOBUILD) -o $(HOMEDIR)/Pinyin-go

# make test, test your code
test: prepare test-case
test-case:
	$(GOTEST) -v -cover $(GOPKGS)

# make package
package: package-bin
package-bin:
	mkdir -p $(OUTDIR)
	mv Pinyin-go  $(OUTDIR)/

# make clean
clean:
	$(GO) clean
	rm -rf $(OUTDIR)
	rm -rf $(HOMEDIR)/Pinyin-go
	rm -rf $(GOPATH)/pkg/darwin_amd64

# avoid filename conflict and speed up build 
.PHONY: all prepare compile test package clean build
