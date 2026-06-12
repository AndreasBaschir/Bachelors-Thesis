TEX_FILE=$(shell ls *.tex)
PDF_FILE=$(TEX_FILE:.tex=.pdf)

.PHONY: all clean
all: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE)
	pdflatex -synctex=1 -interaction=nonstopmode $<
	$(MAKE) clean

clean:
	find . -name "*.aux" -o -name "*.bbl" -o -name "*.blg" -o -name "*.fdb_latexmk" \
	       -o -name "*.fls" -o -name "*.idx" -o -name "*.ilg" -o -name "*.ind" \
	       -o -name "*.lof" -o -name "*.log" -o -name "*.lot" -o -name "*.nlo" \
	       -o -name "*.out" -o -name "*.synctex.gz" -o -name "*.toc" -o -name "*~" \
	       -o -name "*.thm" -o -name "*.dvi" -o -name "*.ps" | xargs rm -f


