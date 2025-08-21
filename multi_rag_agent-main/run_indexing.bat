@echo off
call conda activate arag_env

REM For processing a specific page range (5-8)
python index_pdfs.py --path "data/HR Analytics Notes.pdf" --chunk-size 150 --overlap 25 --batch-size 2 --start-page 5 --end-page 8

REM For processing the entire document (comment out if using page range)
REM python index_pdfs.py --path "data/HR Analytics Notes.pdf" --chunk-size 150 --overlap 25 --batch-size 2

echo Indexing complete!
pause 