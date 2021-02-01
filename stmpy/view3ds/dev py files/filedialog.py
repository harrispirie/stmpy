import tkinter, tkFileDialog
import pylab

def ask_for_config_file():

    print("Please provide location of configuration file.")

    root = tkinter.Tk()
    root.withdraw()
    file_path = tkFileDialog.askopenfilename()
    root.destroy()

    return file_path


def main():

    config_file_path = ask_for_config_file()

    pylab.figure()
    pylab.show()

    print("Made it.")