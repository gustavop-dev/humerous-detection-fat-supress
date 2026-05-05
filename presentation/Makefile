MAIN = main
LATEX = pdflatex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex figures/*.png
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb $(MAIN).pdf

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null
