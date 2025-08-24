import sys
from PyQt5.QtWidgets import QAbstractScrollArea,QApplication,QTextEdit, QWidget, QSizePolicy, QVBoxLayout, QLabel, QRadioButton, QGridLayout, QGroupBox,QPushButton, QLineEdit,QMessageBox , QStackedWidget,QTableWidget,QHeaderView,QTableWidgetItem 
from PyQt5.QtGui import QFont
import sympy as sp
from PyQt5.QtCore import Qt
import re

#method inputs 
class BisectionInputs(QWidget):
    def __init__(self):
        super().__init__()
        layout=QVBoxLayout()
        self.function_label=QLabel('Enter Function f(x):')
        self.function_input=QLineEdit()
        
        layout.addWidget(self.function_label)
        layout.addWidget(self.function_input)

        self.xl_label=QLabel('Enter Xl:')
        self.xl_input=QLineEdit()
        layout.addWidget(self.xl_label)
        layout.addWidget(self.xl_input)

        self.xu_label=QLabel('Enter Xu:')
        self.xu_input=QLineEdit()
        layout.addWidget(self.xu_label)
        layout.addWidget(self.xu_input)

        self.error_label=QLabel('Enter Error Percentage:')
        self.error_input=QLineEdit()
        layout.addWidget(self.error_label)
        layout.addWidget(self.error_input)

        self.setLayout(layout)
        self.setStyleSheet("font-size:14pt")

class fixedpointInputs(QWidget):
    def __init__(self):
        super().__init__()

        layout=QVBoxLayout()

        self.function_label=QLabel("Enter Function f(x):")
        self.function_input=QLineEdit()
        layout.addWidget(self.function_label)
        layout.addWidget(self.function_input)

        self.xi_label=QLabel('Enter Xi:')
        self.xi_input=QLineEdit()
        layout.addWidget(self.xi_label)
        layout.addWidget(self.xi_input)

        self.error_label=QLabel('Enter Error Percentage:')
        self.error_input=QLineEdit()
        layout.addWidget(self.error_label)
        layout.addWidget(self.error_input)

        self.setLayout(layout)
        self.setStyleSheet("font-size:14pt")

class NewtonInputs(QWidget):
    def __init__(self):
        super().__init__()
        layout=QVBoxLayout()
        self.function_label=QLabel('Enter Function f(x):')
        self.function_input=QLineEdit()
        layout.addWidget(self.function_label)
        layout.addWidget(self.function_input)

        self.xi_label=QLabel('Enter X0:')
        self.xi_input=QLineEdit()
        layout.addWidget(self.xi_label)
        layout.addWidget(self.xi_input)

        self.error_label=QLabel('Enter Error Percentage:')
        self.error_input=QLineEdit()
        layout.addWidget(self.error_label)
        layout.addWidget(self.error_input)

        self.setLayout(layout)
        self.setStyleSheet("font-size:14pt")

class SecantInputs(QWidget):
       
    def __init__(self):
       super().__init__()
       layout=QVBoxLayout()
       self.function_label=QLabel('Enter Function f(x):')
       self.function_input=QLineEdit()
            
       layout.addWidget(self.function_label)
       layout.addWidget(self.function_input)

       self.xiOld_label=QLabel('Enter X-1:')
       self.xiOld_input=QLineEdit()
       layout.addWidget(self.xiOld_label)
       layout.addWidget(self.xiOld_input)

       self.xi_label=QLabel('Enter X0:')
       self.xi_input=QLineEdit()
       layout.addWidget(self.xi_label)
       layout.addWidget(self.xi_input)

       self.error_label=QLabel('Enter Error Percentage:')
       self.error_input=QLineEdit()
       layout.addWidget(self.error_label)
       layout.addWidget(self.error_input)

       self.setLayout(layout)
       self.setStyleSheet("font-size:14pt")

class MatrixInputs(QWidget):
     def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        self.equation_label = QLabel("Enter equations:")
        self.equation_input = QTextEdit()

        layout.addWidget(self.equation_label)
        layout.addWidget(self.equation_input)
 
        self.setLayout(layout)
        self.setStyleSheet("font-size:14pt")

#output window 
class OutputWindow(QWidget):
     def __init__(self):
          
          super().__init__()
          
          self.setWindowTitle("Iterations Results")
          self.layout=QVBoxLayout()
          self.setLayout(self.layout)
          
          self.table=QTableWidget()
          self.table.setEditTriggers(QTableWidget.NoEditTriggers) #make the table read only 
           # Make the table resize with the window
          self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
          self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
          self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
          self.root_label = QLabel("")
          self.root_label.setAlignment(Qt.AlignCenter)
          self.root_label.setStyleSheet("font-size: 16pt; font-weight: bold; color: #282E67; padding-top: 10px;")

          self.extra_info_label = QLabel("")
          self.extra_info_label.setWordWrap(True)
          self.extra_info_label.setStyleSheet("font-size: 13pt; color: #444; padding: 10px;")
          self.layout.addWidget(self.extra_info_label)

          self.layout.addWidget(self.table)
          self.layout.addWidget(self.root_label)
          self.table.setStyleSheet("""
                QTableWidget {
                    font-size: 14pt;
                    background-color: #f9f9f9;
                    border: 1px solid #ccc;
                }
                QHeaderView::section {
                    background-color: #282E67;
                    color: white;
                    font-weight: bold;
                    padding: 5px;
                    border: 1px solid #ddd;
                }
                QTableWidget::item {
                    padding: 5px;
                }
            """)
          
          self.method_name = None
          self.data = None
     def set_data(self, data, method_name):
            self.data = data
            self.method_name = method_name
            self.display_results(data, method_name)

     def on_show_clicked(self):
            if self.data and self.method_name:
                self.display_results(self.data, self.method_name)

     def set_extra_info(self, text):
             self.extra_info_label.setText(text)
          
     def display_results(self, data , method_name):
                    
                self.table.setRowCount(len(data))
                self.table.setColumnCount(len(data[0]))

                headers={
               "bisection":['Iter','Xl','F(Xl)','Xu','F(Xu)','Xr','F(Xr)','Error Percentage'],
               "Newton": ['Iter', 'Xn', 'f(Xn)', "f'(Xn)", 'Error Percentage'],
               "Secant": ['Iter', 'Xn-1', 'Xn', 'f(Xn-1)', 'f(Xn)', 'Error Percentage'],
               "Fixed-Point": ['Iter', 'Xn', 'g(Xn)', 'Error Percent'],
               "False-Position": ['Iter','Xl','F(Xl)','Xu','F(Xu)','Xr','F(Xr)','Error Percentage']
                } 
                self.table.setHorizontalHeaderLabels(headers[method_name])
                for row_idx, row in enumerate(data):
                  for col_idx, value in enumerate(row):
                   if col_idx==0:
                       item= QTableWidgetItem(str(int(value)))
                   else:
                        item=QTableWidgetItem( f"{value:.4f}" if isinstance(value, (int, float)) else str(value))

                   item.setTextAlignment(Qt.AlignCenter)
                   self.table.setItem(row_idx, col_idx, item)

                self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                self.table.resizeColumnsToContents()
                self.table.resizeRowsToContents()
                self.table.verticalHeader().setVisible(False)

                try:
                    final_row = data[-1]
                    if method_name in"Newton":
                        final_root = final_row[1]  # Xn column
                    elif method_name == "Secant":
                        final_root = final_row[2] 
                    elif method_name ==["Fixed-Point", "bisection"]:
                        final_root = final_row[5]  # Xr
                    elif method_name == "False-Position":
                        final_root = final_row[3]  # Xr
                    else:
                        final_root = final_row[1]  # default to second column (Xn)

                    self.root_label.setText(f"Final Root ≈{final_root:.6f}")
                except Exception as e:
                    self.root_label.setText("Could not determine final root.")

#matrix output window 
class MatrixOutput(QWidget):
    def __init__(self):
            super().__init__()
            self.setWindowTitle('Matrix Method Output')
            self.resize(800, 600)

            layout = QVBoxLayout()
            self.output_display = QTextEdit()
            self.output_display.setReadOnly(True)

            self.output_display.setFontFamily("Courier New")
            self.output_display.setFontPointSize(12)
            self.output_display.setStyleSheet("""
                QTextEdit {
                    background-color: #ffffff;
                    border: 2px solid #d1d5db;
                    border-radius: 10px;
                    padding: 14px;
                    color: #1e293b;
                }
            """)

            layout.addWidget(self.output_display)
            self.setLayout(layout)

            self.setStyleSheet("""
                QWidget {
                    background-color: #f5f7fa;
                    font-family: 'Segoe UI', Arial;
                    font-size: 14pt;
                }
            """)

    def format_section_title(self, title):
            return f'<span style="color:#4A148C; font-weight:bold; font-size:20pt;">{title}</span><br>'

    def format_step_text(self, text):
            return f'<span style="color:#1A237E;"><b>{text}</b></span><br>'

    def format_matrix(self, matrix):
            return f'<pre>{sp.pretty(matrix)}</pre>'

        # Append formatted content 
    def append_title(self, title):
            self.output_display.append(self.format_section_title(title))

    def append_step(self, text):
            self.output_display.append(self.format_step_text(text))

    def append_matrix(self, matrix):
            self.output_display.append(self.format_matrix(matrix))

    def append_line(self, text):
            self.output_display.append(text)


#Main window (choose method , inputs )
class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle('Numerical Methods')
        self.setGeometry(350,45,650, 550)  # Set the window size
        
        # Create a main vertical layout
        main_layout = QVBoxLayout()

        # Create a label for the header
        self.groupbox= QGroupBox('Select Method:')
        self.groupbox.setFont(QFont('Arial', 16, QFont.Bold))  # Set font size, family, and weight
        self.groupbox.setStyleSheet("color: #333;")  # Set text color
        main_layout.addWidget(self.groupbox)  # Add the label to the top of the layout

        # intialize stak to hold different options for inputs based on the selected method
        self.stack=QStackedWidget() 
        self.bisection_inputs=BisectionInputs()
        self.newton_inputs=NewtonInputs()
        self.fixedpoint_inputs=fixedpointInputs()
        self.secant_inputs=SecantInputs()
        self.Matrix_inputs=MatrixInputs()
        self.stack.addWidget(self.bisection_inputs)
        self.stack.addWidget(self.newton_inputs)
        self.stack.addWidget(self.secant_inputs)
        self.stack.addWidget(self.fixedpoint_inputs)
        self.stack.addWidget(self.Matrix_inputs)
        
        
        # Create a grid layout for the radio buttons
        radio_layout = QGridLayout()

        # Create radio buttons        
        
        self.radio1 = QRadioButton('Bisection Method', self)
        self.radio1.toggled.connect(lambda:self.stack.setCurrentIndex(0))
        self.radio2 = QRadioButton('False Position Method', self)
        self.radio2.toggled.connect(lambda:self.stack.setCurrentIndex(0))
        self.radio3 = QRadioButton('Newton-Raphson Method', self)
        self.radio3.toggled.connect(lambda:self.stack.setCurrentIndex(1))
        self.radio4 = QRadioButton('Secant Method', self)
        self.radio4.toggled.connect(lambda:self.stack.setCurrentIndex(2))
        self.radio5 = QRadioButton('Simple Fixed-Point', self)
        self.radio5.toggled.connect(lambda:self.stack.setCurrentIndex(3))
        self.radio6=QRadioButton('Gauss Elimination Method',self)
        self.radio6.toggled.connect(lambda:self.stack.setCurrentIndex(4))
        self.radio7=QRadioButton('LU decomposition Method',self)
        self.radio7.toggled.connect(lambda:self.stack.setCurrentIndex(4))
        self.radio8=QRadioButton('LU decomposition with partial pivoting ',self)
        self.radio8.toggled.connect(lambda:self.stack.setCurrentIndex(4))
        self.radio9=QRadioButton('Gauss Elimination with partial pivoting ',self)
        self.radio9.toggled.connect(lambda:self.stack.setCurrentIndex(4))
        self.radio10=QRadioButton('Gauss Jorden Method ',self)
        self.radio10.toggled.connect(lambda:self.stack.setCurrentIndex(4))
        self.radio11=QRadioButton('Gauss Jorden with partial pivoting',self)
        self.radio11.toggled.connect(lambda:self.stack.setCurrentIndex(4))
        self.radio12=QRadioButton("cramer's Rule",self)
        self.radio12.toggled.connect(lambda:self.stack.setCurrentIndex(4))
        

        # Add radio buttons to the grid layout in two columns
        radio_layout.addWidget(self.radio1, 0, 0)  # Row 0, Column 0
        radio_layout.addWidget(self.radio2, 0, 1)  
        radio_layout.addWidget(self.radio3, 1, 0) 
        radio_layout.addWidget(self.radio4, 1, 1)  
        radio_layout.addWidget(self.radio5, 2, 0) 
        radio_layout.addWidget(self.radio6, 2, 1) 
        radio_layout.addWidget(self.radio7, 0, 2) 
        radio_layout.addWidget(self.radio8, 1, 2) 
        radio_layout.addWidget(self.radio9, 2, 2) 
        radio_layout.addWidget(self.radio10, 0, 3) 
        radio_layout.addWidget(self.radio11, 1, 3)
        radio_layout.addWidget(self.radio12, 2, 3)
        # Add the radio layout to the main layout
        self.groupbox.setLayout(radio_layout)

        # Set the main layout
        self.setLayout(main_layout)

        # Apply a style sheet to the window
        # self.setStyleSheet("""
        #     QWidget {
        #         background-color: #F0F0F0;
        #         font-size:14pt
        #     }
        #     QLabel{
        #         padding:20px
        #                    }
        # """)

        main_layout.addWidget(self.stack)
        self.calculate_button=QPushButton('Calculate')
        self.calculate_button.clicked.connect(self.calculate)
        main_layout.addWidget(self.calculate_button)
        # self.setStyleSheet("font-size:14pt")

        
    #calculate button handling 
    def calculate(self):

        self.results_window = OutputWindow()
        self.Matrix_output=MatrixOutput()

        if self.radio1.isChecked():
            self.bisection(self.results_window)
        elif self.radio2.isChecked():
            self.falseposition(self.results_window)
        elif self.radio3.isChecked():
            self.newton(self.results_window)
        elif self.radio4.isChecked():
            self.secant(self.results_window)
        elif self.radio5.isChecked():
            self.fixedpoint(self.results_window)
        elif self.radio6.isChecked():
            self.GuassElimination(self.Matrix_output)
        elif self.radio7.isChecked():
            self.LU(self.Matrix_output)
        elif self.radio8.isChecked():
            self.LU_partialPivoting(self.Matrix_output)
        elif self.radio9.isChecked():
           self. GuassElimination_Withppivoting(self.Matrix_output)
        elif self.radio10.isChecked():
            self.gaussJorden(self.Matrix_output)
        elif self.radio11.isChecked():
            self.gaussJorden_withppivoting(self.Matrix_output)
        elif self.radio12.isChecked():
            self.cramer_rule(self.Matrix_output)

        else:
            QMessageBox.warning(self, 'Method Error', 'Please select a method!')
            return
        
   #bisecton method
    def bisection(self ,results_window):

        # get inputs from BisectionInputs widget
        func=self.bisection_inputs.function_input.text()
        xl=self.bisection_inputs.xl_input.text()
        xu=self.bisection_inputs.xu_input.text()
        error=self.bisection_inputs.error_input.text()

        # validate inputs
        if not all([xu,xl,func,error]):
            QMessageBox.warning(self,'invalid input','all fields are required!!!')

        try:
                # make x as unknown variable
                x=sp.Symbol('x')
                #  convert the string(function) into a mathematical expression 
                f=sp.lambdify(x,sp.sympify(func))

                xu = float(xu)
                xl = float(xl)
                error = float(error)
                # show a error message if the user entered a nun numeric input
        except ValueError:
                QMessageBox.warning(self, 'Input Error', 'All inputs must be valid numbers!')
                return
        
        if f(xl)*f(xu)>=0:
            raise ValueError('Invalid initial guesses — f(xl) and f(xu) must have opposite signs.')    
            
        iteration=0 
        xr=0
        output=[]
        e=float('inf')
        fxl=f(xl)
        fxu=f(xu)
        while e>error :
            xrOld=xr
            
            xr=(xl+xu)/2
            fxr=f(xr)

            if iteration > 0:
             e=abs((xr-xrOld)/xr)*100 

            output.append([iteration,xl,fxl,xu,fxu,xr,fxr,e if iteration >0 else'-'])
            
            if fxl * fxr < 0:
             xu = xr
             fxu = fxr
            else:
             xl = xr
             fxl = fxr

            iteration += 1

        results_window.set_data(output,"bisection") 
        results_window.show()

   #false-position method 
    def falseposition(self,results_window):
    # get inputs from BisectionInputs widget
        func=self.bisection_inputs.function_input.text()
        xl=self.bisection_inputs.xl_input.text()
        xu=self.bisection_inputs.xu_input.text()
        error=self.bisection_inputs.error_input.text()

        # validate inputs
        if not all([xu,xl,func,error]):
            QMessageBox.warning(self,'invalid input','all fields are required!!!')

        try:
                # make x as unknown variable
                x=sp.Symbol('x')
                #  convert the string(function) into a mathematical expression 
                f=sp.lambdify(x,sp.sympify(func))

                xu = float(xu)
                xl = float(xl)
                error = float(error)
                # show a error message if the user entered a nun numeric input
        except ValueError:
                QMessageBox.warning(self, 'Input Error', 'All inputs must be valid numbers!')
                return
        
        if f(xl)*f(xu)>=0:
            raise ValueError('Invalid initial guesses — f(xl) and f(xu) must have opposite signs.')  
        
        iteration=0 
        xr=0
        output=[]
        e=float('inf')
        fxl=f(xl)
        fxu=f(xu)
        while e>error :
            xrOld=xr
            
            xr=xu-((f(xu)*(xl-xu))/(f(xl)-f(xu)))
            fxr=f(xr)

            if iteration > 0:
             e=abs((xr-xrOld)/xr)*100 

            output.append([iteration,xl,fxl,xu,fxu,xr,fxr,e if iteration >0 else'-'])
            
            if fxl * fxr < 0:
             xu = xr
             fxu = fxr
            else:
             xl = xr
             fxl = fxr

            iteration += 1
        results_window.set_data(output,"False-Position") 
        results_window.show()
 
    #newton method 
    def newton(self,results_window):

        func=self.newton_inputs.function_input.text()
        xi=float(self.newton_inputs.xi_input.text())
        error=float(self.newton_inputs.error_input.text())

        x=sp.Symbol('x')
        sf=sp.sympify(func)
        f=sp.lambdify(x,sf)
        sdf=sp.diff(sf,x)
        df=sp.lambdify(x,sdf)

        iteration = 0
        output = []
        e = float('inf')
        while e > error:
            fxi = f(xi)
            fxi_deriv = df(xi)
            xi_plus1 = xi - fxi/fxi_deriv
            
            output.append([iteration,xi,fxi,fxi_deriv,e if iteration>0 else '-'])
            e = abs((xi_plus1 - xi)/xi_plus1) * 100

            xi = xi_plus1
            iteration += 1

        output.append([iteration,xi,fxi,fxi_deriv,e])
        results_window.set_extra_info(f"<b>f(x) = {func}</b><br><b>f'(x) = {sdf}</b>")
        results_window.set_data(output,"Newton") 

        results_window.show()

   # secant method 
    def secant(self,results_window):
        func=self.secant_inputs.function_input.text()
        xiOld=float(self.secant_inputs.xiOld_input.text())
        xi=float(self.secant_inputs.xi_input.text())
        error=float(self.secant_inputs.error_input.text())
        x=sp.Symbol('x')
        f=sp.lambdify(x,sp.sympify(func))
        iteration=0
        output=[]
        e=float('inf')
        
        while e>error:
            fxi=f(xi)
            fxiOld=f(xiOld)
            
            xiNew=xi-(fxi*(xiOld-xi))/(fxiOld-fxi)
            output.append([iteration,xiOld,fxiOld,xi,fxi,e if iteration>0 else '-'])
            e=abs((xiNew - xi)/xiNew) * 100
            
            xiOld=xi
            xi=xiNew
            iteration+=1
        output.append([iteration,xiOld,fxiOld,xi,fxi,e])
        results_window.set_data(output,"Secant") 
        results_window.show()

  #fixed point method 
    def fixedpoint (self,results_window):

        func =self.fixedpoint_inputs.function_input.text()
        xi=float(self.fixedpoint_inputs.xi_input.text())
        error=float(self.fixedpoint_inputs.error_input.text())

        function=func.replace('x','y',count=1) #replace the first x in the equation with y 

        x=sp.Symbol('x')
        y=sp.Symbol('y')

        f=sp.sympify(function)
        sg=sp.solve(sp.Eq(f,0),y) #solve the equation to y 
        g=sp.lambdify(x,sg[1]) #get the second output or the positive one 
        
        output=[]
        iteration=0
        e=float('inf')
        while e>error:
            gx=g(xi)
            xiNew=gx
            
            output.append([iteration ,xi,gx,e if iteration>0 else '-'])
            e=abs((xiNew-xi)/xiNew)*100
            
            xi=xiNew
            iteration+=1
        output.append([iteration ,xi,gx,e if iteration>0 else '-'])
        results_window.set_extra_info(f"<b>Simple Fixed-Point= {sg[1]}")
        results_window.set_data(output,"Fixed-Point") 
        results_window.show()

   # gauss elimmination method 
    def GuassElimination(self, Matrix_output): 
        
        A, b,symbols_list= self.get_matrix()
        matrix = A.row_join(b)
        mat = matrix.copy().as_mutable()
        n = mat.rows

        Matrix_output.append_title("-Augmented Matrix")
        Matrix_output.append_matrix(matrix)


        # Elimination
        for i in range(n):
            for j in range(i + 1, n):
                if mat[i, i] == 0:
                    raise ValueError(f"Error!! Zero pivot at row {i + 1}")

                multiplier = mat[j, i] / mat[i, i]

                for k in range(i, mat.cols):
                    mat[j, k] = sp.simplify(mat[j, k] - multiplier * mat[i, k])

               
                Matrix_output.append_step(f"Operation: R{j + 1} → R{j + 1} - ({multiplier}) * R{i + 1}")
                Matrix_output.append_matrix(mat)

        # Back Substitution
        vals = [0 for _ in range(n)]

        for i in reversed(range(n)):
            sum_ax = sum(mat[i, j] * vals[j] for j in range(i + 1, n))
            vals[i] = sp.simplify((mat[i, -1] - sum_ax) / mat[i, i])

        Matrix_output.append_title("-Final Solution")
        for i, val in enumerate(vals):
           Matrix_output.output_display.append(f"<b>{symbols_list[i]}</b> = {val}")

        Matrix_output.show()

    #LU method 
    def LU(self, Matrix_output):

        A, b,symbols_list= self.get_matrix()
        A = A.copy().as_mutable()
        b = b.copy().as_mutable() #make matrix editable 
        n = A.rows
        
        Matrix_output.append_title("Matrix A:\n")
        Matrix_output.append_matrix(A)
        Matrix_output.append_title("Matrix b:\n")
        Matrix_output.append_matrix(b)

        L = sp.eye(n)
        U = A.copy()

        for i in range(n):
            for j in range(i + 1, n):
                if U[i, i] == 0:
                    raise ValueError(f"Error!! Zero pivot at row {i + 1}")
                
                multiplier = U[j, i] / U[i, i]
                L[j, i] = multiplier

                for k in range(i, n):
                    U[j, k] = sp.simplify(U[j, k] - multiplier * U[i, k])

                Matrix_output.append_step( f"\nOperation: R{j + 1} → R{j + 1} - ({multiplier}) * R{i + 1}\n" )
                Matrix_output.append_matrix(U)

        # Step 1: Forward substitution to solve Lc = b
        
        c = sp.zeros(n, 1)
        for i in range(n):
            sum_lc = sum(L[i, j] * c[j] for j in range(i))
            c[i] = sp.simplify((b[i] - sum_lc) / L[i, i])

        # Step 2: Backward substitution to solve Ux = c
        x = sp.zeros(n, 1)
        for i in reversed(range(n)):
            sum_ux = sum(U[i, j] * x[j] for j in range(i + 1, n))
            x[i] = sp.simplify((c[i] - sum_ux) / U[i, i])

        # Output results
        Matrix_output.append_step("<b>Lower Triangular Matrix L:</b>" )
        Matrix_output.append_matrix(L) 
        Matrix_output.append_step("<b>Upper Triangular Matrix U:</b>" )
        Matrix_output.append_matrix(U) 
        Matrix_output.append_title("Step 1: Solve Lc = b ")
        Matrix_output.append_step("c =")
        Matrix_output.append_matrix(c) 
        Matrix_output.append_title("Step 2: Solve Ux = c ")
        Matrix_output.append_step("x =")
        Matrix_output.append_matrix(x)
        Matrix_output.append_title("Final Solution :")
        for i, val in enumerate(x):
            Matrix_output.output_display.append(f"<b>{symbols_list[i]}</b> = {val}")
        Matrix_output.show()

  # LU with partial pivoting 
    def LU_partialPivoting(self, Matrix_output):
        A, b,symbols_list= self.get_matrix()
        A = A.as_mutable()
        b = b.as_mutable()
        n = A.rows
        output=[]
        # Output initial matrices
        Matrix_output.append_title("Matrix A:\n")
        Matrix_output.append_matrix(A)
        Matrix_output.append_title("Matrix b:\n")
        Matrix_output.append_matrix(b)

        # Initialize L and U
        L = sp.eye(n)
        U = A.copy()

        for i in range(n):
            # Partial pivoting
            max_row = max(range(i, n), key=lambda r: abs(U[r, i]))
            if U[max_row, i] == 0:
                raise ValueError(f"Zero pivot at column {i + 1}")

            if max_row != i:
                U.row_swap(i, max_row)
                b.row_swap(i, max_row)

                # Swap L rows only before the current column
                for k in range(i):
                    L[i, k], L[max_row, k] = L[max_row, k], L[i, k]

                Matrix_output.append_step(f"<b>Pivoting:</b> Swap R{i + 1} with R{max_row + 1}")
                Matrix_output.append_matrix(U)

            # Elimination
            for j in range(i + 1, n):
                multiplier = U[j, i] / U[i, i]
                L[j, i] = multiplier
                for k in range(i, n):
                    U[j, k] = sp.simplify(U[j, k] - multiplier * U[i, k])

                Matrix_output.append_step(f"<b>Operation:</b> R{j + 1} → R{j + 1} - ({multiplier}) * R{i + 1}")
                Matrix_output.append_matrix(U)

        # Forward substitution to solve Lc = b
        c = sp.zeros(n, 1)
        for i in range(n):
            sum_lc = sum(L[i, j] * c[j] for j in range(i))
            c[i] = sp.simplify((b[i] - sum_lc) / L[i, i])

        # Backward substitution to solve Ux = c
        x = sp.zeros(n, 1)
        for i in reversed(range(n)):
            sum_ux = sum(U[i, j] * x[j] for j in range(i + 1, n))
            x[i] = sp.simplify((c[i] - sum_ux) / U[i, i])

        # Output results
        Matrix_output.append_step("<b>Lower Triangular Matrix L:</b>" )
        Matrix_output.append_matrix(L) 
        Matrix_output.append_step("<b>Upper Triangular Matrix U:</b>" )
        Matrix_output.append_matrix(U) 
        Matrix_output.append_title("Step 1: Solve Lc = b ")
        Matrix_output.append_step("c =")
        Matrix_output.append_matrix(c) 
        Matrix_output.append_title("Step 2: Solve Ux = c ")
        Matrix_output.append_step("x =")
        Matrix_output.append_matrix(x)
        Matrix_output.append_title("Final Solution :")

        for i, val in enumerate(x):
           Matrix_output.output_display.append(f"<b>{symbols_list[i]}</b> = {val}")

        Matrix_output.show()

  # Gauss elimination with partial pivoting 
    def GuassElimination_Withppivoting(self, Matrix_output):
        A, b,symbols_list= self.get_matrix()
        matrix = A.row_join(b)
        mat = matrix.copy().as_mutable()
        n = mat.rows

        Matrix_output.append_title("Augmented Matrix:\n")
        Matrix_output.append_matrix(matrix)
        # Forward Elimination
        for i in range(n):
            # Partial pivoting
            max_row = max(range(i, n), key=lambda r: abs(mat[r, i]))
            if mat[max_row, i] == 0:
                raise ValueError(f"Zero pivot encountered at column {i + 1}")

            if max_row != i:
                mat.row_swap(i, max_row)

                Matrix_output.append_step(f"<b>Pivoting:</b> Swap R{i + 1} with R{max_row + 1}")
                Matrix_output.append_matrix(mat)

            for j in range(i + 1, n):
                if mat[i, i] == 0:
                    raise ValueError(f"Error!! Zero pivot at row {i + 1}")

                multiplier = mat[j, i] / mat[i, i]

                for k in range(i, mat.cols):
                    mat[j, k] = sp.simplify(mat[j, k] - multiplier * mat[i, k])

                Matrix_output.append_step(f"\n Operation: R{j + 1} → R{j + 1} - ({multiplier}) * R{i + 1}\n")

                Matrix_output.append_matrix(mat)

        # Back Substitution
        vals = [0 for _ in range(n)]

        for i in reversed(range(n)):
            sum_ax = sum(mat[i, j] * vals[j] for j in range(i + 1, n))
            vals[i] = sp.simplify((mat[i, -1] - sum_ax) / mat[i, i])

        Matrix_output.append_title("Final Solution:\n")
        for i, val in enumerate(vals):
           Matrix_output.output_display.append(f"<b>{symbols_list[i]}</b> = {val}")

        Matrix_output.show()

  # Gauss jorden 
    def gaussJorden(self,Matrix_output):

        A, b ,symbols_list= self.get_matrix()
        matrix = A.row_join(b)
        mat = matrix.copy().as_mutable()
        n = mat.rows

        Matrix_output.append_title("Augmented Matrix:\n")
        Matrix_output.append_matrix(matrix)
        
        for i in range(n):
            pivot = mat[i, i]
            if pivot==0:
                raise ValueError(f"Zero pivot encountered at column {i + 1}")
            # Make the pivot for each col equal 1 by dividing the whole row by that pivot.
            for k in range(i,mat.cols):
                mat[i,k]=sp.simplify(mat[i, k] / pivot)
            
            Matrix_output.append_title(f"\nNormalize Row {i + 1} (R{i + 1} / {pivot}):\n")
            Matrix_output.append_matrix(mat)

            for j in range (n):
                if j !=i : # to avoid zeroing thr pivot no 
                    factor =mat[j,i]
                    for k in range(i,mat.cols):
                        mat[j, k] = sp.simplify(mat[j, k] - factor * mat[i, k])

                    Matrix_output.append_step( f"\nOperation: R{j + 1} → R{j + 1} - ({factor}) * R{i + 1}\n")
                    Matrix_output.append_matrix(mat) 

        # Final solution
        Matrix_output.append_title("Final Solution:")
        for i in range(n):
            Matrix_output.output_display.append(f"<b>{symbols_list[i]}</b> = {mat[i, -1]}")

        Matrix_output.show()
        
  # gauss jorden with partial piovting 
    def gaussJorden_withppivoting(self, Matrix_output):

        A, b, symbols_list = self.get_matrix()
        matrix = A.row_join(b)
        mat = matrix.copy().as_mutable()
        n = mat.rows

        Matrix_output.append_title("Augmented Matrix:\n")
        Matrix_output.append_matrix(matrix)

        # Gauss-Jordan Elimination with Partial Pivoting
        for i in range(n):
            # Partial Pivoting: find the row with the max absolute value in column i
            max_row = i
            max_val = abs(mat[i, i])
            for r in range(i + 1, n):
                if abs(mat[r, i]) > max_val:
                    max_val = abs(mat[r, i])
                    max_row = r

            if max_val == 0:
                raise ValueError(f"Zero pivot encountered in column {i + 1}")

            # Swap rows if needed
            if max_row != i:
                mat.row_swap(i, max_row)
                Matrix_output.append_step(f"<b>Pivoting:</b> Swap R{i + 1} with R{max_row + 1}")
                Matrix_output.append_matrix(mat)

            # Normalize pivot row
            pivot = mat[i, i]
            for k in range(i, mat.cols):
                mat[i, k] = sp.simplify(mat[i, k] / pivot)

            Matrix_output.append_title(f"\nNormalize Row {i + 1} (R{i + 1} / {pivot}):\n")
            Matrix_output.append_matrix(mat)

            # Eliminate other entries in column i
            for j in range(n):
                if j != i:
                    factor = mat[j, i]
                    for k in range(i, mat.cols):
                        mat[j, k] = sp.simplify(mat[j, k] - factor * mat[i, k])
                    Matrix_output.append_step( f"\nOperation: R{j + 1} → R{j + 1} - ({factor}) * R{i + 1}\n")
                    Matrix_output.append_matrix(mat) 

        # Final solution
        Matrix_output.append_title("Final Solution:")
        for i in range(n):
            Matrix_output.output_display.append(f"<b>{symbols_list[i]}</b> = {mat[i, -1]}")

        Matrix_output.show()

    #cramer's rule 
    def cramer_rule(self, Matrix_output):

        A, b, vars = self.get_matrix()
        detA=A.det()
        Matrix_output.append_title('Original Coefficient matrix A:')
        Matrix_output.append_matrix(A)
        Matrix_output.append_step(f"Det(A)={detA}")
        result=[]
        for i in range(len(vars)):
            Ai=A.copy()
            Ai[:,i]=b   #replace the column i with b 
            detAi=Ai.det()
            xi=detAi/detA
            result.append(xi)

            Matrix_output.append_step(f" Matrix A{i+1} replace column {i+1}with b:")
            Matrix_output.append_matrix(Ai)
            Matrix_output.output_display.append(f"<b>Det(A{i+1})</b> = {detAi} ->  x{i+1} = {detAi}/{detA} = <b>{xi}</b><br><br>")
        
        Matrix_output.append_title("Final Solution:")

        for i,val in enumerate(result):
            Matrix_output.output_display.append(f"x{i+1}=<b>{val}</b><br>")
        
        Matrix_output.show()


    def get_matrix(self):

        # Get equations from input
        equ_text=self.Matrix_inputs.equation_input.toPlainText().strip()
        lines=equ_text.splitlines()

        # extract variable names from equations 
        all_vars=sorted(set(re.findall(r'[a-zA-Z_]\w*',equ_text)))
        vars=sp.symbols(all_vars)  

        equations=[]
        for line in lines:
            lhs,rhs=line.split('=')
            equations.append(sp.Eq(sp.simplify(rhs),sp.simplify(lhs)))

        #get A and b matrices   
        A,b=sp.linear_eq_to_matrix(equations,vars)

        return A,b ,vars 
    
   #styling
STYLESHEET = """
    /* Main Window */
    QWidget {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', Arial;
        font-size: 13px;
    }
    
    QGroupBox {
        border: 2px solid #d1d5db;
        border-radius: 8px;
        margin-top: 15px;
        padding-top: 15px;
        padding-bottom: 15px;
        background-color: white;
    }
    
    /* Input Fields */
    QLineEdit, QTextEdit {
        border: 1px solid #AAAECD ;
        padding: 8px 12px;
        background-color: white;
        selection-background-color: #7c3aed;
        border-radius: 4px;
    }
    
    QLineEdit:focus, QTextEdit:focus {
        border: 2px solid #AAAECD;
        outline: none;
    }
    /*  Output */
    QTextEdit {
        background-color: white;
        border-radius: 6px;
        padding: 12px;
    }
    QPushButton {
        background-color:#282E67;
        color: white;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        
    }
    QPushButton:hover {
        background-color: #AAAECD;
    }
    

    QRadioButton {
        font-size: 10pt;
        color: #333;
        padding: 30px;
      }
      
    QRadioButton:hover {       
        color:#282E67;
    }
"""

# Create the application object
app = QApplication(sys.argv)
# Apply style
app.setStyleSheet(STYLESHEET)
# Create the window object
window = MyWindow()
window.show()

# Execute the application
sys.exit(app.exec_())