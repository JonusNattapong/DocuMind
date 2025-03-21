# Sample Document Information

## Creating a Test Document

For testing DocuMind, you'll need a sample PDF document. Here are instructions for creating a simple test document:

### Option 1: Create an Invoice PDF

You can create a simple invoice PDF using Microsoft Word, Google Docs, or any office software. Include the following elements:

1. **Invoice Header**:
   - Invoice Number: INV-12345
   - Date: 21 March 2025
   - Customer: บริษัท ตัวอย่าง จำกัด

2. **Invoice Items (Table)**:
   - Column Headers: รายการ | จำนวน | ราคาต่อหน่วย | ภาษีมูลค่าเพิ่ม | รวม
   - Row 1: สินค้า A | 2 | ฿1,000.00 | ฿140.00 | ฿2,140.00
   - Row 2: สินค้า B | 3 | ฿500.00 | ฿105.00 | ฿1,605.00
   - Row 3: สินค้า C | 1 | ฿3,000.00 | ฿210.00 | ฿3,210.00

3. **Invoice Footer**:
   - รวม: ฿4,500.00
   - ภาษีมูลค่าเพิ่ม (7%): ฿455.00
   - ยอดรวมสุทธิ: ฿6,955.00

Save this document as `sample_document.pdf` in the project root directory.

### Option 2: Download a Sample Document

Alternatively, you can download sample invoice PDFs from various online resources and rename them to `sample_document.pdf`.

## Testing with the Sample Document

After creating or downloading the sample document, you can test DocuMind using:

```
python example.py
```

For testing with a local model (without Google API):

```
python example.py --local
```

If you have a GPU:

```
python example.py --gpu
```
