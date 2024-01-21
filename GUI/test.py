from PyQt5.QtWidgets import QLabel

# ... inside MainWindow.__init__

self.test_label1 = QLabel("Panel 1")
self.test_label2 = QLabel("Panel 2")
self.layout.addWidget(self.test_label1)
self.layout.addWidget(self.test_label2)

