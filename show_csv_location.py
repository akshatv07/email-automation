import os
import glob

def show_csv_location():
    """Show where CSV files are saved and list existing ones"""
    print("📁 CSV File Location Information")
    print("=" * 50)
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"📂 Current working directory: {current_dir}")
    
    # Find all CSV files in current directory
    csv_files = glob.glob("*.csv")
    
    if csv_files:
        print(f"\n📄 Found {len(csv_files)} CSV file(s) in current directory:")
        total_size = 0
        for csv_file in sorted(csv_files):
            file_path = os.path.join(current_dir, csv_file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            print(f"   📄 {csv_file} ({file_size:,} bytes)")
        
        print(f"\n📊 Total size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
        
        # Show file details
        print(f"\n🔍 File details:")
        for csv_file in sorted(csv_files):
            file_path = os.path.join(current_dir, csv_file)
            file_size = os.path.getsize(file_path)
            modified_time = os.path.getmtime(file_path)
            from datetime import datetime
            modified_date = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   📄 {csv_file}")
            print(f"      Size: {file_size:,} bytes")
            print(f"      Modified: {modified_date}")
            print(f"      Full path: {file_path}")
            print()
    else:
        print("\n❌ No CSV files found in current directory")
        print("💡 Run print_coll.py or print_coll_simple.py to generate CSV files")
    
    # Show how to open files
    print("💡 To open CSV files:")
    print("   - Double-click the .csv file")
    print("   - Or open with Excel/LibreOffice")
    print("   - Or use: start filename.csv (Windows)")
    print("   - Or use: open filename.csv (Mac)")

if __name__ == "__main__":
    show_csv_location() 