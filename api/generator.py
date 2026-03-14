import time
import subprocess

def generate_hu_number() -> str:
    prefix = "10000010"
    timestamp_part = str(int(time.time() * 100))[-10:]
    hu = f"{prefix}{timestamp_part}"
    result = hu[:18].ljust(18, '0')
    assert len(result) == 18, f"Generated HU should be 18 chars, got {len(result)}"
    return result

def generate_ibd_number() -> str:
    prefix = "IBD"
    timestamp_part = str(int(time.time() * 1000))[-10:]
    return f"{prefix}{timestamp_part}"

def generate_datamatrix_string(hu, part, vendor_lot, scanned_qty, msd_level, vendor_code) -> str:
    return f"RID{hu}PRN{part}LOT{vendor_lot}QTY{scanned_qty}MSD{msd_level}VEN{vendor_code}"

def generate_zpl(datamatrix_string, ibd, part, description, qty, uom, hu, vendor, invoice, lot, msd_level, msd_date) -> str:
    desc_short = (description[:37] + '...') if len(description) > 40 else description
    
    zpl = f"""^XA
^FX DataMatrix Barcode
^FO50,50^BXN,10,200,0,0,1,~
^FD{datamatrix_string}^FS

^FX Text Info Right Side
^FO350,50^A0N,30,30^FDIBD No# : {ibd}^FS
^FO350,90^A0N,30,30^FDPart : {part}^FS
^FO350,130^A0N,30,30^FD{desc_short}^FS
^FO350,170^A0N,30,30^FDQTY : {qty} / {qty} {uom}^FS

^FX Divider
^FO50,220^GB700,3,3^FS

^FX Bottom Info
^FO300,240^A0N,40,40^FD{hu}^FS
^FO50,300^A0N,30,30^FDVendor : {vendor}^FS
^FO50,340^A0N,30,30^FDSupplier Invoice : {invoice}^FS
^FO50,380^A0N,30,30^FDVen Lot No : {lot}^FS
^FO50,420^A0N,30,30^FDMSD Level : {msd_level}   MSD Date : {msd_date}^FS
^XZ"""
    return zpl

def print_label(zpl_code):
    import platform
    try:
        with open("/tmp/label.zpl", "w") as f:
            f.write(zpl_code)

        if platform.system() == "Windows":
            # Windows print command
            subprocess.run([
                "powershell", "-Command",
                f'Add-PrinterPort -Name "FILE:" -PrinterPortType File; Print-Object -FilePath /tmp/label.zpl -PrinterName "zebra_printer"'
            ], check=False)
        else:
            # Unix/Linux print command
            subprocess.run([
                "lp",
                "-d",
                "zebra_printer",
                "/tmp/label.zpl"
            ], check=True)
        return True
    except Exception as e:
        print(f"Print failed: {e}")
        return False
