# SAP EWM Training PDFs

Place your SAP EWM training documentation PDFs in this directory.

## Supported File Types
- PDF files (`.pdf` extension)

## Recommended Documents
- SAP EWM User Guides
- SAP EWM Configuration Manuals  
- SAP EWM Troubleshooting Guides
- SAP EWM Best Practices
- SAP EWM Process Documentation
- Custom warehouse procedures and SOPs

## File Naming Recommendations
- Use descriptive filenames
- Avoid special characters and spaces
- Examples:
  - `SAP_EWM_User_Guide_v2.pdf`
  - `EWM_Troubleshooting_Manual.pdf`
  - `Warehouse_Operations_SOP.pdf`

## Important Notes
- Ensure you have proper licensing/rights to use the documentation
- PDFs will be indexed with full content and metadata
- Larger files may take longer to process during indexing
- The system will automatically detect and skip duplicate files (based on content hash)

## After Adding PDFs
Run the indexing script to make them available to Skippy:

```bash
python build_index.py
```
