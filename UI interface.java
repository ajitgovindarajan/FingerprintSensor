import javax.swing.*;
import javax.swing.border.TitledBorder;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.ArrayList;
import java.text.SimpleDateFormat;
import java.util.Date;

public class FingerprintRecognitionUI extends JFrame {
    
    // Core system components
    private FingerprintRecognitionSystem fingerprintSystem;
    
    // UI Components
    private JTabbedPane mainTabbedPane;
    private JPanel enrollmentPanel;
    private JPanel recognitionPanel;
    private JPanel managementPanel;
    private JPanel settingsPanel;
    
    // Enrollment components
    private JTextField userIdField;
    private JLabel fingerprintImageLabel;
    private JProgressBar enrollmentProgressBar;
    private JTextArea enrollmentLogArea;
    private JButton captureButton;
    private JButton enrollButton;
    private JButton clearEnrollmentButton;
    private JLabel qualityLabel;
    private JLabel livenessLabel;
    
    // Recognition components
    private JLabel recognitionImageLabel;
    private JButton recognizeButton;
    private JTextArea recognitionResultArea;
    private JLabel matchStatusLabel;
    private JLabel confidenceLabel;
    private JProgressBar recognitionProgressBar;
    
    // Management components
    private JTable userTable;
    private DefaultTableModel tableModel;
    private JButton deleteUserButton;
    private JButton viewTemplateButton;
    private JTextField searchField;
    private JLabel totalUsersLabel;
    
    // Settings components
    private JSlider qualityThresholdSlider;
    private JSlider confidenceThresholdSlider;
    private JComboBox<String> sensorTypeCombo;
    private JCheckBox livenessDetectionCheckbox;
    private JSpinner enrollmentSamplesSpinner;
    
    // Status components
    private JLabel statusLabel;
    private JLabel systemModeLabel;
    
    // Current captured images
    private BufferedImage currentRawImage;
    private BufferedImage currentProcessedImage;
    
    public FingerprintRecognitionUI() {
        initializeSystem();
        initializeUI();
        setupEventHandlers();
    }
    
    private void initializeSystem() {
        try {
            fingerprintSystem = new FingerprintRecognitionSystem();
            updateStatus("System initialized successfully", false);
        } catch (Exception e) {
            updateStatus("Failed to initialize system: " + e.getMessage(), true);
        }
    }
    
    private void initializeUI() {
        setTitle("Fingerprint Recognition System v2.0");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1200, 800);
        setLocationRelativeTo(null);
        setResizable(true);
        
        // Set Look and Feel
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeel());
        } catch (Exception e) {
            // Use default look and feel
        }
        
        createComponents();
        layoutComponents();
        updateStatus("UI initialized", false);
    }
    
    private void createComponents() {
        // Main tabbed pane
        mainTabbedPane = new JTabbedPane();
        
        // Create panels
        createEnrollmentPanel();
        createRecognitionPanel();
        createManagementPanel();
        createSettingsPanel();
        
        // Status bar
        createStatusBar();
        
        // Add tabs
        mainTabbedPane.addTab("Enrollment", new ImageIcon(), enrollmentPanel, "Enroll new users");
        mainTabbedPane.addTab("Recognition", new ImageIcon(), recognitionPanel, "Recognize fingerprints");
        mainTabbedPane.addTab("Management", new ImageIcon(), managementPanel, "Manage enrolled users");
        mainTabbedPane.addTab("Settings", new ImageIcon(), settingsPanel, "System configuration");
    }
    
    private void createEnrollmentPanel() {
        enrollmentPanel = new JPanel(new BorderLayout());
        
        // Left panel - User input and controls
        JPanel leftPanel = new JPanel(new GridBagLayout());
        leftPanel.setBorder(new TitledBorder("User Information"));
        GridBagConstraints gbc = new GridBagConstraints();
        
        // User ID input
        gbc.gridx = 0; gbc.gridy = 0; gbc.anchor = GridBagConstraints.WEST;
        leftPanel.add(new JLabel("User ID:"), gbc);
        
        gbc.gridx = 1; gbc.gridy = 0; gbc.fill = GridBagConstraints.HORIZONTAL; gbc.weightx = 1.0;
        userIdField = new JTextField(20);
        leftPanel.add(userIdField, gbc);
        
        // Buttons
        gbc.gridx = 0; gbc.gridy = 1; gbc.gridwidth = 2; gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(10, 0, 5, 0);
        captureButton = new JButton("Capture Fingerprint");
        captureButton.setPreferredSize(new Dimension(200, 30));
        leftPanel.add(captureButton, gbc);
        
        gbc.gridy = 2; gbc.insets = new Insets(5, 0, 5, 0);
        enrollButton = new JButton("Enroll User");
        enrollButton.setEnabled(false);
        leftPanel.add(enrollButton, gbc);
        
        gbc.gridy = 3;
        clearEnrollmentButton = new JButton("Clear");
        leftPanel.add(clearEnrollmentButton, gbc);
        
        // Quality and liveness indicators
        gbc.gridy = 4; gbc.insets = new Insets(20, 0, 5, 0);
        qualityLabel = new JLabel("Image Quality: Not captured");
        leftPanel.add(qualityLabel, gbc);
        
        gbc.gridy = 5; gbc.insets = new Insets(5, 0, 5, 0);
        livenessLabel = new JLabel("Liveness Status: Not tested");
        leftPanel.add(livenessLabel, gbc);
        
        // Progress bar
        gbc.gridy = 6; gbc.insets = new Insets(10, 0, 5, 0);
        enrollmentProgressBar = new JProgressBar(0, 100);
        enrollmentProgressBar.setStringPainted(true);
        leftPanel.add(enrollmentProgressBar, gbc);
        
        // Center panel - Fingerprint image display
        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.setBorder(new TitledBorder("Fingerprint Image"));
        
        fingerprintImageLabel = new JLabel();
        fingerprintImageLabel.setPreferredSize(new Dimension(300, 300));
        fingerprintImageLabel.setHorizontalAlignment(JLabel.CENTER);
        fingerprintImageLabel.setBorder(BorderFactory.createLoweredBevelBorder());
        fingerprintImageLabel.setText("No image captured");
        
        JScrollPane imageScrollPane = new JScrollPane(fingerprintImageLabel);
        centerPanel.add(imageScrollPane, BorderLayout.CENTER);
        
        // Right panel - Log area
        JPanel rightPanel = new JPanel(new BorderLayout());
        rightPanel.setBorder(new TitledBorder("Enrollment Log"));
        
        enrollmentLogArea = new JTextArea(15, 25);
        enrollmentLogArea.setEditable(false);
        enrollmentLogArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JScrollPane logScrollPane = new JScrollPane(enrollmentLogArea);
        rightPanel.add(logScrollPane, BorderLayout.CENTER);
        
        // Add panels to enrollment panel
        enrollmentPanel.add(leftPanel, BorderLayout.WEST);
        enrollmentPanel.add(centerPanel, BorderLayout.CENTER);
        enrollmentPanel.add(rightPanel, BorderLayout.EAST);
    }
    
    private void createRecognitionPanel() {
        recognitionPanel = new JPanel(new BorderLayout());
        
        // Left panel - Recognition controls
        JPanel leftPanel = new JPanel(new GridBagLayout());
        leftPanel.setBorder(new TitledBorder("Recognition Controls"));
        GridBagConstraints gbc = new GridBagConstraints();
        
        // Recognize button
        gbc.gridx = 0; gbc.gridy = 0; gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.insets = new Insets(10, 10, 10, 10);
        recognizeButton = new JButton("Start Recognition");
        recognizeButton.setPreferredSize(new Dimension(200, 40));
        leftPanel.add(recognizeButton, gbc);
        
        // Status indicators
        gbc.gridy = 1; gbc.insets = new Insets(20, 10, 5, 10);
        matchStatusLabel = new JLabel("Status: Ready");
        leftPanel.add(matchStatusLabel, gbc);
        
        gbc.gridy = 2; gbc.insets = new Insets(5, 10, 5, 10);
        confidenceLabel = new JLabel("Confidence: --");
        leftPanel.add(confidenceLabel, gbc);
        
        // Progress bar
        gbc.gridy = 3; gbc.insets = new Insets(10, 10, 10, 10);
        recognitionProgressBar = new JProgressBar(0, 100);
        recognitionProgressBar.setStringPainted(true);
        leftPanel.add(recognitionProgressBar, gbc);
        
        // Center panel - Image display
        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.setBorder(new TitledBorder("Live Fingerprint"));
        
        recognitionImageLabel = new JLabel();
        recognitionImageLabel.setPreferredSize(new Dimension(400, 400));
        recognitionImageLabel.setHorizontalAlignment(JLabel.CENTER);
        recognitionImageLabel.setBorder(BorderFactory.createLoweredBevelBorder());
        recognitionImageLabel.setText("Place finger on sensor");
        
        JScrollPane recImageScrollPane = new JScrollPane(recognitionImageLabel);
        centerPanel.add(recImageScrollPane, BorderLayout.CENTER);
        
        // Right panel - Results
        JPanel rightPanel = new JPanel(new BorderLayout());
        rightPanel.setBorder(new TitledBorder("Recognition Results"));
        
        recognitionResultArea = new JTextArea(20, 30);
        recognitionResultArea.setEditable(false);
        recognitionResultArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        JScrollPane resultScrollPane = new JScrollPane(recognitionResultArea);
        rightPanel.add(resultScrollPane, BorderLayout.CENTER);
        
        // Add panels
        recognitionPanel.add(leftPanel, BorderLayout.WEST);
        recognitionPanel.add(centerPanel, BorderLayout.CENTER);
        recognitionPanel.add(rightPanel, BorderLayout.EAST);
    }
    
    private void createManagementPanel() {
        managementPanel = new JPanel(new BorderLayout());
        
        // Top panel - Search and controls
        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        topPanel.add(new JLabel("Search:"));
        searchField = new JTextField(20);
        topPanel.add(searchField);
        
        JButton searchButton = new JButton("Search");
        topPanel.add(searchButton);
        
        JButton refreshButton = new JButton("Refresh");
        topPanel.add(refreshButton);
        
        totalUsersLabel = new JLabel("Total Users: 0");
        topPanel.add(totalUsersLabel);
        
        // Center panel - User table
        String[] columnNames = {"User ID", "Enrollment Date", "Template Quality", "Last Access"};
        tableModel = new DefaultTableModel(columnNames, 0) {
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;
            }
        };
        
        userTable = new JTable(tableModel);
        userTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        userTable.setRowHeight(25);
        JScrollPane tableScrollPane = new JScrollPane(userTable);
        
        // Bottom panel - Management buttons
        JPanel bottomPanel = new JPanel(new FlowLayout());
        deleteUserButton = new JButton("Delete User");
        deleteUserButton.setEnabled(false);
        bottomPanel.add(deleteUserButton);
        
        viewTemplateButton = new JButton("View Template");
        viewTemplateButton.setEnabled(false);
        bottomPanel.add(viewTemplateButton);
        
        JButton exportButton = new JButton("Export Data");
        bottomPanel.add(exportButton);
        
        // Add panels
        managementPanel.add(topPanel, BorderLayout.NORTH);
        managementPanel.add(tableScrollPane, BorderLayout.CENTER);
        managementPanel.add(bottomPanel, BorderLayout.SOUTH);
    }
    
    private void createSettingsPanel() {
        settingsPanel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        
        // System Settings
        JPanel systemPanel = new JPanel(new GridBagLayout());
        systemPanel.setBorder(new TitledBorder("System Settings"));
        GridBagConstraints sysgbc = new GridBagConstraints();
        
        // Sensor Type
        sysgbc.gridx = 0; sysgbc.gridy = 0; sysgbc.anchor = GridBagConstraints.WEST;
        sysgbc.insets = new Insets(5, 5, 5, 5);
        systemPanel.add(new JLabel("Sensor Type:"), sysgbc);
        
        sysgbc.gridx = 1; sysgbc.fill = GridBagConstraints.HORIZONTAL;
        sensorTypeCombo = new JComboBox<>(new String[]{"Optical", "Capacitive", "Ultrasonic"});
        systemPanel.add(sensorTypeCombo, sysgbc);
        
        // Quality Threshold
        sysgbc.gridx = 0; sysgbc.gridy = 1; sysgbc.fill = GridBagConstraints.NONE;
        systemPanel.add(new JLabel("Quality Threshold:"), sysgbc);
        
        sysgbc.gridx = 1; sysgbc.fill = GridBagConstraints.HORIZONTAL;
        qualityThresholdSlider = new JSlider(0, 100, 60);
        qualityThresholdSlider.setMajorTickSpacing(20);
        qualityThresholdSlider.setPaintTicks(true);
        qualityThresholdSlider.setPaintLabels(true);
        systemPanel.add(qualityThresholdSlider, sysgbc);
        
        // Confidence Threshold
        sysgbc.gridx = 0; sysgbc.gridy = 2; sysgbc.fill = GridBagConstraints.NONE;
        systemPanel.add(new JLabel("Match Confidence:"), sysgbc);
        
        sysgbc.gridx = 1; sysgbc.fill = GridBagConstraints.HORIZONTAL;
        confidenceThresholdSlider = new JSlider(0, 100, 80);
        confidenceThresholdSlider.setMajorTickSpacing(20);
        confidenceThresholdSlider.setPaintTicks(true);
        confidenceThresholdSlider.setPaintLabels(true);
        systemPanel.add(confidenceThresholdSlider, sysgbc);
        
        // Liveness Detection
        sysgbc.gridx = 0; sysgbc.gridy = 3; sysgbc.gridwidth = 2;
        livenessDetectionCheckbox = new JCheckBox("Enable Liveness Detection");
        livenessDetectionCheckbox.setSelected(true);
        systemPanel.add(livenessDetectionCheckbox, sysgbc);
        
        // Enrollment Samples
        sysgbc.gridx = 0; sysgbc.gridy = 4; sysgbc.gridwidth = 1; sysgbc.fill = GridBagConstraints.NONE;
        systemPanel.add(new JLabel("Enrollment Samples:"), sysgbc);
        
        sysgbc.gridx = 1; sysgbc.fill = GridBagConstraints.HORIZONTAL;
        enrollmentSamplesSpinner = new JSpinner(new SpinnerNumberModel(3, 1, 5, 1));
        systemPanel.add(enrollmentSamplesSpinner, sysgbc);
        
        // Buttons Panel
        JPanel buttonPanel = new JPanel(new FlowLayout());
        JButton saveSettingsButton = new JButton("Save Settings");
        JButton resetSettingsButton = new JButton("Reset to Defaults");
        JButton testSystemButton = new JButton("Test System");
        
        buttonPanel.add(saveSettingsButton);
        buttonPanel.add(resetSettingsButton);
        buttonPanel.add(testSystemButton);
        
        // Layout settings panel
        gbc.gridx = 0; gbc.gridy = 0; gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 1.0; gbc.weighty = 0.7;
        settingsPanel.add(systemPanel, gbc);
        
        gbc.gridy = 1; gbc.weighty = 0.3; gbc.fill = GridBagConstraints.HORIZONTAL;
        settingsPanel.add(buttonPanel, gbc);
    }
    
    private void createStatusBar() {
        JPanel statusBar = new JPanel(new BorderLayout());
        statusBar.setBorder(BorderFactory.createLoweredBevelBorder());
        
        statusLabel = new JLabel("Ready");
        statusLabel.setBorder(BorderFactory.createEmptyBorder(2, 5, 2, 5));
        
        systemModeLabel = new JLabel("Mode: Standby");
        systemModeLabel.setBorder(BorderFactory.createEmptyBorder(2, 5, 2, 5));
        
        statusBar.add(statusLabel, BorderLayout.WEST);
        statusBar.add(systemModeLabel, BorderLayout.EAST);
        
        add(statusBar, BorderLayout.SOUTH);
    }
    
    private void layoutComponents() {
        setLayout(new BorderLayout());
        add(mainTabbedPane, BorderLayout.CENTER);
    }
    
    private void setupEventHandlers() {
        // Enrollment tab event handlers
        captureButton.addActionListener(e -> captureFingerprint());
        enrollButton.addActionListener(e -> enrollUser());
        clearEnrollmentButton.addActionListener(e -> clearEnrollmentForm());
        
        // Recognition tab event handlers
        recognizeButton.addActionListener(e -> startRecognition());
        
        // Management tab event handlers
        userTable.getSelectionModel().addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                boolean hasSelection = userTable.getSelectedRow() != -1;
                deleteUserButton.setEnabled(hasSelection);
                viewTemplateButton.setEnabled(hasSelection);
            }
        });
        
        deleteUserButton.addActionListener(e -> deleteSelectedUser());
        viewTemplateButton.addActionListener(e -> viewSelectedTemplate());
        
        // Settings tab event handlers
        qualityThresholdSlider.addChangeListener(e -> updateQualityThreshold());
        confidenceThresholdSlider.addChangeListener(e -> updateConfidenceThreshold());
        
        // Tab change handler
        mainTabbedPane.addChangeListener(e -> onTabChanged());
    }
    
    // Event handler methods
    private void captureFingerprint() {
        updateStatus("Capturing fingerprint...", false);
        captureButton.setEnabled(false);
        enrollmentProgressBar.setIndeterminate(true);
        
        // Simulate fingerprint capture in background thread
        SwingWorker<BufferedImage, String> worker = new SwingWorker<BufferedImage, String>() {
            @Override
            protected BufferedImage doInBackground() throws Exception {
                publish("Initializing sensor...");
                Thread.sleep(500);
                
                publish("Waiting for finger placement...");
                Thread.sleep(1000);
                
                publish("Capturing image...");
                BufferedImage capturedImage = fingerprintSystem.captureFingerprintImage();
                Thread.sleep(500);
                
                publish("Analyzing quality...");
                double quality = fingerprintSystem.assessImageQuality(capturedImage);
                Thread.sleep(300);
                
                publish("Testing liveness...");
                boolean isLive = fingerprintSystem.detectLiveness(capturedImage);
                Thread.sleep(300);
                
                // Update UI on EDT
                SwingUtilities.invokeLater(() -> {
                    qualityLabel.setText(String.format("Image Quality: %.1f%%", quality * 100));
                    livenessLabel.setText("Liveness Status: " + (isLive ? "Live" : "Suspicious"));
                    
                    if (quality >= 0.6 && isLive) {
                        enrollButton.setEnabled(true);
                    }
                });
                
                return capturedImage;
            }
            
            @Override
            protected void process(List<String> chunks) {
                for (String message : chunks) {
                    appendToEnrollmentLog(message);
                }
            }
            
            @Override
            protected void done() {
                try {
                    currentRawImage = get();
                    displayFingerprintImage(currentRawImage);
                    appendToEnrollmentLog("Fingerprint captured successfully");
                } catch (Exception e) {
                    appendToEnrollmentLog("Error: " + e.getMessage());
                }
                
                captureButton.setEnabled(true);
                enrollmentProgressBar.setIndeterminate(false);
                enrollmentProgressBar.setValue(0);
                updateStatus("Ready", false);
            }
        };
        
        worker.execute();
    }
    
    private void enrollUser() {
        String userId = userIdField.getText().trim();
        if (userId.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please enter a User ID", "Input Required", 
                                        JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        if (currentRawImage == null) {
            JOptionPane.showMessageDialog(this, "Please capture a fingerprint first", "Image Required", 
                                        JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        updateStatus("Enrolling user: " + userId, false);
        enrollButton.setEnabled(false);
        enrollmentProgressBar.setIndeterminate(true);
        
        SwingWorker<EnrollmentResult, String> worker = new SwingWorker<EnrollmentResult, String>() {
            @Override
            protected EnrollmentResult doInBackground() throws Exception {
                publish("Starting enrollment for " + userId + "...");
                
                for (int i = 1; i <= 3; i++) {
                    publish("Processing sample " + i + "/3...");
                    Thread.sleep(800);
                    
                    SwingUtilities.invokeLater(() -> 
                        enrollmentProgressBar.setValue((int)(33.3 * i)));
                }
                
                publish("Creating master template...");
                Thread.sleep(500);
                
                publish("Storing in database...");
                Thread.sleep(300);
                
                // Simulate enrollment
                EnrollmentResult result = fingerprintSystem.enrollFingerprint(userId);
                return result;
            }
            
            @Override
            protected void process(List<String> chunks) {
                for (String message : chunks) {
                    appendToEnrollmentLog(message);
                }
            }
            
            @Override
            protected void done() {
                try {
                    EnrollmentResult result = get();
                    if (result.isSuccess()) {
                        appendToEnrollmentLog("✓ Enrollment successful!");
                        JOptionPane.showMessageDialog(FingerprintRecognitionUI.this, 
                            "User enrolled successfully!", "Enrollment Complete", 
                            JOptionPane.INFORMATION_MESSAGE);
                        clearEnrollmentForm();
                        refreshUserTable();
                    } else {
                        appendToEnrollmentLog("✗ Enrollment failed: " + result.getMessage());
                        JOptionPane.showMessageDialog(FingerprintRecognitionUI.this, 
                            "Enrollment failed: " + result.getMessage(), "Enrollment Error", 
                            JOptionPane.ERROR_MESSAGE);
                    }
                } catch (Exception e) {
                    appendToEnrollmentLog("✗ Error: " + e.getMessage());
                }
                
                enrollButton.setEnabled(true);
                enrollmentProgressBar.setIndeterminate(false);
                enrollmentProgressBar.setValue(100);
                updateStatus("Ready", false);
            }
        };
        
        worker.execute();
    }
    
    private void startRecognition() {
        updateStatus("Starting recognition...", false);
        recognizeButton.setEnabled(false);
        recognitionProgressBar.setIndeterminate(true);
        matchStatusLabel.setText("Status: Scanning...");
        
        SwingWorker<MatchResult, String> worker = new SwingWorker<MatchResult, String>() {
            @Override
            protected MatchResult doInBackground() throws Exception {
                publish("Waiting for fingerprint...");
                Thread.sleep(1000);
                
                publish("Capturing image...");
                Thread.sleep(500);
                
                publish("Processing image...");
                Thread.sleep(800);
                
                publish("Extracting features...");
                Thread.sleep(600);
                
                publish("Matching against database...");
                Thread.sleep(1000);
                
                // Simulate recognition
                MatchResult result = fingerprintSystem.recognizeFingerprint();
                return result;
            }
            
            @Override
            protected void process(List<String> chunks) {
                for (String message : chunks) {
                    appendToRecognitionLog(message);
                }
            }
            
            @Override
            protected void done() {
                try {
                    MatchResult result = get();
                    displayRecognitionResult(result);
                } catch (Exception e) {
                    appendToRecognitionLog("Error: " + e.getMessage());
                    matchStatusLabel.setText("Status: Error");
                }
                
                recognizeButton.setEnabled(true);
                recognitionProgressBar.setIndeterminate(false);
                recognitionProgressBar.setValue(100);
                updateStatus("Ready", false);
                
                // Reset after 3 seconds
                Timer timer = new Timer(3000, e -> {
                    recognitionProgressBar.setValue(0);
                    matchStatusLabel.setText("Status: Ready");
                    confidenceLabel.setText("Confidence: --");
                });
                timer.setRepeats(false);
                timer.start();
            }
        };
        
        worker.execute();
    }
    
    // Helper methods
    private void displayFingerprintImage(BufferedImage image) {
        if (image != null) {
            // Scale image to fit label
            int labelWidth = fingerprintImageLabel.getWidth();
            int labelHeight = fingerprintImageLabel.getHeight();
            
            if (labelWidth > 0 && labelHeight > 0) {
                Image scaledImage = image.getScaledInstance(labelWidth - 10, labelHeight - 10, Image.SCALE_SMOOTH);
                fingerprintImageLabel.setIcon(new ImageIcon(scaledImage));
                fingerprintImageLabel.setText("");
            }
        }
    }
    
    private void appendToEnrollmentLog(String message) {
        SwingUtilities.invokeLater(() -> {
            String timestamp = new SimpleDateFormat("HH:mm:ss").format(new Date());
            enrollmentLogArea.append("[" + timestamp + "] " + message + "\n");
            enrollmentLogArea.setCaretPosition(enrollmentLogArea.getDocument().getLength());
        });
    }
    
    private void appendToRecognitionLog(String message) {
        SwingUtilities.invokeLater(() -> {
            String timestamp = new SimpleDateFormat("HH:mm:ss").format(new Date());
            recognitionResultArea.append("[" + timestamp + "] " + message + "\n");
            recognitionResultArea.setCaretPosition(recognitionResultArea.getDocument().getLength());
        });
    }
    
    private void displayRecognitionResult(MatchResult result) {
        if (result.isMatch()) {
            matchStatusLabel.setText("Status: MATCH FOUND");
            matchStatusLabel.setForeground(Color.GREEN);
            confidenceLabel.setText(String.format("Confidence: %.1f%%", result.getConfidence() * 100));
            appendToRecognitionLog("✓ Match found - User: " + result.getUserId());
            appendToRecognitionLog("Confidence: " + String.format("%.2f%%", result.getConfidence() * 100));
        } else {
            matchStatusLabel.setText("Status: NO MATCH");
            matchStatusLabel.setForeground(Color.RED);
            confidenceLabel.setText("Confidence: --");
            appendToRecognitionLog("✗ No match found");
        }
    }
    
    private void clearEnrollmentForm() {
        userIdField.setText("");
        fingerprintImageLabel.setIcon(null);
        fingerprintImageLabel.setText("No image captured");
        qualityLabel.setText("Image Quality: Not captured");
        livenessLabel.setText("Liveness Status: Not tested");
        enrollmentLogArea.setText("");
        enrollButton.setEnabled(false);
        currentRawImage = null;
        currentProcessedImage = null;
    }
    
    private void refreshUserTable() {
        // Clear existing data
        tableModel.setRowCount(0);
        
        // Simulate loading user data
        // In real implementation, this would query the database
        Object[][] sampleData = {
            {"user001", "2024-01-15", "High", "2024-01-20"},
            {"user002", "2024-01-16", "Medium", "2024-01-19"},
            {"admin", "2024-01-10", "High", "2024-01-21"}
        };
        
        for (Object[] row : sampleData) {
            tableModel.addRow(row);
        }
        
        totalUsersLabel.setText("Total Users: " + tableModel.getRowCount());
    }
    
    private void deleteSelectedUser() {
        int selectedRow = userTable.getSelectedRow();
        if (selectedRow != -1) {
            String userId = (String) tableModel.getValueAt(selectedRow, 0);
            int confirm = JOptionPane.showConfirmDialog(this,
                "Are you sure you want to delete user: " + userId + "?",
                "Confirm Delete", JOptionPane.YES_NO_OPTION);
                
            if (confirm == JOptionPane.YES_OPTION) {
                tableModel.removeRow(selectedRow);
                totalUsersLabel.setText("Total Users: " + tableModel.getRowCount());
                updateStatus("User " + userId + " deleted", false);
            }
        }
    }
    
    private void viewSelectedTemplate() {
        int selectedRow = userTable.getSelectedRow();
        if (selectedRow != -1) {
            String userId = (String) tableModel.getValueAt(selectedRow, 0);
            
            // Create template viewer dialog
            JDialog templateDialog = new JDialog(this, "Template Details - " + userId, true);
            templateDialog.setSize(600, 500);
            templateDialog.setLocationRelativeTo(this);
            
            JPanel contentPanel = new JPanel(new BorderLayout());
            
            // Template info
            JTextArea templateInfo = new JTextArea(20, 50);
            templateInfo.setEditable(false);
            templateInfo.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            templateInfo.setText(generateTemplateInfo(userId));
            
            JScrollPane scrollPane = new JScrollPane(templateInfo);
            contentPanel.add(scrollPane, BorderLayout.CENTER);
            
            JPanel buttonPanel = new JPanel(new FlowLayout());
            JButton closeButton = new JButton("Close");
            closeButton.addActionListener(e -> templateDialog.dispose());
            buttonPanel.add(closeButton);
            
            contentPanel.add(buttonPanel, BorderLayout.SOUTH);
            templateDialog.add(contentPanel);
            templateDialog.setVisible(true);
        }
    }
    
    private String generateTemplateInfo(String userId) {
        StringBuilder info = new StringBuilder();
        info.append("=== FINGERPRINT TEMPLATE DETAILS ===\n");
        info.append("User ID: ").append(userId).append("\n");
        info.append("Enrollment Date: 2024-01-15 10:30:25\n");
        info.append("Template Version: 2.0\n");
        info.append("Quality Score: 0.87\n\n");
        
        info.append("=== MINUTIAE FEATURES ===\n");
        info.append("Total Minutiae: 45\n");
        info.append("Ridge Endings: 28\n");
        info.append("Bifurcations: 17\n\n");
        
        info.append("Sample Minutiae Points:\n");
        info.append("1. Type: ENDING, X: 125, Y: 87, Angle: 45.2°\n");
        info.append("2. Type: BIFURCATION, X: 156, Y: 134, Angle: 122.8°\n");
        info.append("3. Type: ENDING, X: 98, Y: 201, Angle: 78.5°\n");
        info.append("4. Type: BIFURCATION, X: 203, Y: 165, Angle: 201.3°\n");
        info.append("...\n\n");
        
        info.append("=== AI FEATURES ===\n");
        info.append("Deep Feature Vector Size: 256\n");
        info.append("Feature Extraction Model: CNN-v2.1\n");
        info.append("Texture Features: 64 dimensions\n");
        info.append("Ridge Pattern Score: 0.91\n\n");
        
        info.append("=== MATCHING STATISTICS ===\n");
        info.append("Successful Matches: 23\n");
        info.append("Failed Attempts: 1\n");
        info.append("Average Match Confidence: 0.94\n");
        info.append("Last Access: 2024-01-21 14:25:10\n");
        
        return info.toString();
    }
    
    private void updateQualityThreshold() {
        double threshold = qualityThresholdSlider.getValue() / 100.0;
        // Update system threshold
        updateStatus("Quality threshold set to " + (int)(threshold * 100) + "%", false);
    }
    
    private void updateConfidenceThreshold() {
        double threshold = confidenceThresholdSlider.getValue() / 100.0;
        // Update system threshold
        updateStatus("Confidence threshold set to " + (int)(threshold * 100) + "%", false);
    }
    
    private void onTabChanged() {
        int selectedTab = mainTabbedPane.getSelectedIndex();
        switch (selectedTab) {
            case 0: // Enrollment
                systemModeLabel.setText("Mode: Enrollment");
                break;
            case 1: // Recognition
                systemModeLabel.setText("Mode: Recognition");
                break;
            case 2: // Management
                systemModeLabel.setText("Mode: Management");
                refreshUserTable();
                break;
            case 3: // Settings
                systemModeLabel.setText("Mode: Settings");
                break;
        }
    }
    
    private void updateStatus(String message, boolean isError) {
        SwingUtilities.invokeLater(() -> {
            statusLabel.setText(message);
            statusLabel.setForeground(isError ? Color.RED : Color.BLACK);
        });
    }
    
    // Main method and application startup
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                // Set system look and feel
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeel());
                
                // Create and display the application
                FingerprintRecognitionUI app = new FingerprintRecognitionUI();
                app.setVisible(true);
                
                // Show startup dialog
                showStartupDialog(app);
                
            } catch (Exception e) {
                JOptionPane.showMessageDialog(null,
                    "Failed to start application: " + e.getMessage(),
                    "Startup Error",
                    JOptionPane.ERROR_MESSAGE);
                System.exit(1);
            }
        });
    }
    
    private static void showStartupDialog(JFrame parent) {
        JDialog startupDialog = new JDialog(parent, "System Initialization", true);
        startupDialog.setSize(400, 200);
        startupDialog.setLocationRelativeTo(parent);
        startupDialog.setDefaultCloseOperation(JDialog.DO_NOTHING_ON_CLOSE);
        
        JPanel contentPanel = new JPanel(new BorderLayout());
        
        JLabel messageLabel = new JLabel("Initializing Fingerprint Recognition System...", JLabel.CENTER);
        messageLabel.setFont(messageLabel.getFont().deriveFont(Font.BOLD, 14f));
        
        JProgressBar progressBar = new JProgressBar();
        progressBar.setIndeterminate(true);
        progressBar.setStringPainted(true);
        progressBar.setString("Loading AI models and hardware drivers...");
        
        contentPanel.add(messageLabel, BorderLayout.NORTH);
        contentPanel.add(progressBar, BorderLayout.CENTER);
        
        startupDialog.add(contentPanel);
        startupDialog.setVisible(true);
        
        // Simulate initialization process
        Timer initTimer = new Timer(3000, e -> {
            startupDialog.dispose();
            JOptionPane.showMessageDialog(parent,
                "System initialized successfully!\nReady for fingerprint operations.",
                "Initialization Complete",
                JOptionPane.INFORMATION_MESSAGE);
        });
        initTimer.setRepeats(false);
        initTimer.start();
    }
}

// Supporting classes for the UI (these would typically be in separate files)

class Point2D {
    private int x, y;
    
    public Point2D(int x, int y) {
        this.x = x;
        this.y = y;
    }
    
    public int getX() { return x; }
    public int getY() { return y; }
}

class RidgePattern {
    private String patternType;
    private double[] characteristics;
    
    public RidgePattern(String patternType, double[] characteristics) {
        this.patternType = patternType;
        this.characteristics = characteristics;
    }
    
    public String getPatternType() { return patternType; }
    public double[] getCharacteristics() { return characteristics; }
}

class TextureFeatures {
    private double[] features;
    
    public TextureFeatures(double[] features) {
        this.features = features;
    }
    
    public double[] getFeatures() { return features; }
}

// Dummy implementations for demonstration (in real app, these would connect to actual system)
class SensorInterface {
    public enum SensorType { OPTICAL, CAPACITIVE, ULTRASONIC }
    
    private SensorType sensorType = SensorType.OPTICAL;
    
    public SensorType getSensorType() { return sensorType; }
    public void setSensorType(SensorType type) { this.sensorType = type; }
    
    public BufferedImage captureOpticalImage() {
        return createDummyFingerprintImage();
    }
    
    public BufferedImage captureCapacitiveImage() {
        return createDummyFingerprintImage();
    }
    
    public BufferedImage captureUltrasonicImage() {
        return createDummyFingerprintImage();
    }
    
    private BufferedImage createDummyFingerprintImage() {
        BufferedImage image = new BufferedImage(300, 300, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = image.createGraphics();
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, 300, 300);
        g2d.setColor(Color.BLACK);
        g2d.setStroke(new BasicStroke(2));
        
        // Draw some curved lines to simulate fingerprint ridges
        for (int i = 0; i < 20; i++) {
            int y = 15 + i * 13;
            g2d.drawArc(50 + i * 2, y, 200 - i * 3, 50, 0, 180);
        }
        
        g2d.dispose();
        return image;
    }
}

class ImageProcessor {
    public BufferedImage applyGaussianFilter(BufferedImage image, double sigma) {
        // Dummy implementation - return original image
        return image;
    }
    
    public BufferedImage normalizeIntensity(BufferedImage image) {
        return image;
    }
    
    public BufferedImage enhanceRidges(BufferedImage image) {
        return image;
    }
    
    public BufferedImage convertToBinary(BufferedImage image, int threshold) {
        return image;
    }
    
    public BufferedImage thinRidges(BufferedImage image) {
        return image;
    }
}

class FeatureExtractor {
    // Dummy implementation for demo purposes
}

class DatabaseManager {
    private List<FingerprintTemplate> templates = new ArrayList<>();
    
    public List<FingerprintTemplate> getAllTemplates() {
        return templates;
    }
    
    public boolean storeTemplate(String userId, FingerprintTemplate template) {
        templates.add(template);
        return true;
    }
    
    public boolean deleteTemplate(String userId) {
        return templates.removeIf(t -> t.getUserId().equals(userId));
    }
}

class AIModelManager {
    public NeuralNetworkModel getFeatureExtractionModel() {
        return new NeuralNetworkModel();
    }
    
    public NeuralNetworkModel getSimilarityModel() {
        return new NeuralNetworkModel();
    }
    
    public NeuralNetworkModel getLivenessDetectionModel() {
        return new NeuralNetworkModel();
    }
    
    public NeuralNetworkModel getQualityAssessmentModel() {
        return new NeuralNetworkModel();
    }
}

class NeuralNetworkModel {
    public double[] forward(double[][][] input) {
        // Dummy implementation
        return new double[]{Math.random()};
    }
    
    public double[] forward(double[] input) {
        // Dummy implementation
        return new double[]{Math.random()};
    }
}

class FingerprintTemplate {
    private String userId;
    private List<Minutia> minutiae;
    private double[] deepFeatures;
    private Date enrollmentDate;
    
    public FingerprintTemplate(String userId) {
        this.userId = userId;
        this.enrollmentDate = new Date();
        this.minutiae = new ArrayList<>();
    }
    
    public String getUserId() { return userId; }
    public List<Minutia> getMinutiae() { return minutiae; }
    public double[] getDeepFeatures() { return deepFeatures; }
    public Date getEnrollmentDate() { return enrollmentDate; }
    
    public void setMinutiae(List<Minutia> minutiae) { this.minutiae = minutiae; }
    public void setDeepFeatures(double[] deepFeatures) { this.deepFeatures = deepFeatures; }
}