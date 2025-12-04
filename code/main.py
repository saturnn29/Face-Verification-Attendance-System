import customtkinter as ctk
from attendance_app import AttendanceApp

def main():
    root = ctk.CTk()
    app = AttendanceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
