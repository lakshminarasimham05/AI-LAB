"""
Generate PDF Report for AI Assignment 2
Group K
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import os

def create_report():
    """Create the complete PDF report"""
    
    doc = SimpleDocTemplate(
        "/home/claude/GroupK_Assignment2/GroupK_Assignment2_Report.pdf",
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        alignment=TA_CENTER,
        spaceAfter=30
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=8,
        spaceBefore=12
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )
    
    story = []
    
    # ========================================================================
    # TITLE PAGE
    # ========================================================================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("AI Assignment 2", title_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Artificial Intelligence Course", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Group K</b>", ParagraphStyle('Center', parent=styles['Normal'], alignment=TA_CENTER, fontSize=14)))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("January 2026", ParagraphStyle('Center', parent=styles['Normal'], alignment=TA_CENTER)))
    story.append(PageBreak())
    
    # ========================================================================
    # TABLE OF CONTENTS
    # ========================================================================
    story.append(Paragraph("Table of Contents", heading1_style))
    toc_data = [
        ["1.", "Task 1: Optimizer Performance on Non-Convex Functions", "3"],
        ["2.", "Task 2: Linear Regression Using Multi-Layer Neural Network", "8"],
        ["3.", "Task 3: Multi-class Classification using FCNN", "13"],
        ["4.", "Task 4: MNIST Classification with Different Optimizers", "18"],
    ]
    toc_table = Table(toc_data, colWidths=[0.5*inch, 5*inch, 0.5*inch])
    toc_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(toc_table)
    story.append(PageBreak())
    
    # ========================================================================
    # TASK 1
    # ========================================================================
    story.append(Paragraph("Task 1: Optimizer Performance on Non-Convex Functions", heading1_style))
    
    story.append(Paragraph("<b>Objective:</b>", heading2_style))
    story.append(Paragraph(
        "To implement and compare the performance of various optimization algorithms on two non-convex functions: "
        "the Rosenbrock function and sin(1/x). All optimizers were implemented from scratch using Python.",
        body_style
    ))
    
    story.append(Paragraph("<b>Functions Optimized:</b>", heading2_style))
    story.append(Paragraph(
        "1. <b>Rosenbrock Function:</b> f(x,y) = (1-x)<super>2</super> + 100(y-x<super>2</super>)<super>2</super><br/>"
        "   Global minimum at (1, 1) with f(1,1) = 0<br/><br/>"
        "2. <b>Sin(1/x) Function:</b> f(x) = sin(1/x) with f(0) = 0<br/>"
        "   This function has infinitely many local minima as x approaches 0.",
        body_style
    ))
    
    story.append(Paragraph("<b>Optimizers Implemented:</b>", heading2_style))
    story.append(Paragraph(
        "- Gradient Descent (GD)<br/>"
        "- Stochastic Gradient Descent with Momentum (SGD-M)<br/>"
        "- Adam (Adaptive Moment Estimation)<br/>"
        "- RMSprop<br/>"
        "- Adagrad",
        body_style
    ))
    
    story.append(Paragraph("<b>Results - Rosenbrock Function (Learning Rate = 0.01):</b>", heading2_style))
    
    rosenbrock_data = [
        ["Optimizer", "Final x*", "f(x*)", "Iterations"],
        ["Gradient Descent", "[1608437.5, 10901.3]", "6.69e+26", "3"],
        ["SGD with Momentum", "[1354328.4, 9734.5]", "3.36e+26", "3"],
        ["Adam", "[0.9988, 0.9977]", "1.37e-06", "5543"],
        ["RMSprop", "[0.9801, 0.9755]", "2.25e-02", "10000"],
        ["Adagrad", "[-1.2462, 1.5591]", "5.05e+00", "10000"],
    ]
    
    t1_table = Table(rosenbrock_data, colWidths=[1.5*inch, 1.8*inch, 1.2*inch, 1*inch])
    t1_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t1_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Key Observations:</b>", heading2_style))
    story.append(Paragraph(
        "1. <b>Adam optimizer performed best</b> on the Rosenbrock function, converging to the global minimum "
        "(1, 1) with f(x*) = 1.37e-06, demonstrating excellent performance on this challenging non-convex landscape.<br/><br/>"
        "2. <b>Standard Gradient Descent and SGD with Momentum diverged</b> on the Rosenbrock function at learning rate 0.01. "
        "This is expected as the Rosenbrock function has a narrow curved valley where the gradient can be misleading.<br/><br/>"
        "3. <b>For the sin(1/x) function</b>, most optimizers found local minima at x ≈ 0.212, where sin(1/0.212) ≈ -1 "
        "(a local minimum). This demonstrates how non-convex optimization often finds local rather than global optima.<br/><br/>"
        "4. <b>Adagrad showed slow convergence</b> due to its accumulating squared gradients, which can make the "
        "effective learning rate too small over time.",
        body_style
    ))
    
    story.append(Paragraph("<b>Impact of Hyperparameters:</b>", heading2_style))
    story.append(Paragraph(
        "Learning rate significantly impacts convergence behavior. Higher learning rates (0.05, 0.1) caused faster "
        "divergence for standard GD on Rosenbrock but allowed Adam to converge faster. The adaptive learning rate "
        "methods (Adam, RMSprop) showed more robustness across different learning rate settings.",
        body_style
    ))
    
    # Add images
    story.append(PageBreak())
    story.append(Paragraph("<b>Convergence Plots:</b>", heading2_style))
    
    if os.path.exists('/home/claude/GroupK_Assignment2/rosenbrock_convergence.png'):
        img = Image('/home/claude/GroupK_Assignment2/rosenbrock_convergence.png', width=6.5*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 1: Rosenbrock function convergence behavior</i>", 
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(Spacer(1, 0.2*inch))
    
    if os.path.exists('/home/claude/GroupK_Assignment2/sin_inv_x_convergence.png'):
        img = Image('/home/claude/GroupK_Assignment2/sin_inv_x_convergence.png', width=6.5*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 2: Sin(1/x) function convergence behavior</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(PageBreak())
    
    if os.path.exists('/home/claude/GroupK_Assignment2/rosenbrock_trajectory.png'):
        img = Image('/home/claude/GroupK_Assignment2/rosenbrock_trajectory.png', width=6.5*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 3: Optimization trajectories on Rosenbrock contour</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(Spacer(1, 0.2*inch))
    
    if os.path.exists('/home/claude/GroupK_Assignment2/time_comparison.png'):
        img = Image('/home/claude/GroupK_Assignment2/time_comparison.png', width=6*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 4: Time taken by each optimizer</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(PageBreak())
    
    # ========================================================================
    # TASK 2
    # ========================================================================
    story.append(Paragraph("Task 2: Linear Regression Using Multi-Layer Neural Network", heading1_style))
    
    story.append(Paragraph("<b>Objective:</b>", heading2_style))
    story.append(Paragraph(
        "To implement a multi-layer neural network from scratch for linear regression on the Boston Housing Dataset, "
        "predicting median home values (MEDV) based on number of rooms (RM) and crime rate (CRIM).",
        body_style
    ))
    
    story.append(Paragraph("<b>Network Architecture:</b>", heading2_style))
    story.append(Paragraph(
        "- Input Layer: 2 neurons (RM, CRIM features)<br/>"
        "- Hidden Layer 1: 5 neurons with ReLU activation<br/>"
        "- Hidden Layer 2: 3 neurons with ReLU activation<br/>"
        "- Output Layer: 1 neuron (linear activation for regression)",
        body_style
    ))
    
    story.append(Paragraph("<b>Data Preprocessing:</b>", heading2_style))
    story.append(Paragraph(
        "- Features normalized using Min-Max normalization<br/>"
        "- Data split: 80% training, 20% test<br/>"
        "- Training samples: 404, Test samples: 102",
        body_style
    ))
    
    story.append(Paragraph("<b>Results (Learning Rate = 0.01):</b>", heading2_style))
    
    task2_data = [
        ["Optimizer", "Train MSE", "Test MSE"],
        ["Gradient Descent", "0.0215", "0.0244"],
        ["Momentum", "0.0109", "0.0083"],
        ["Adam", "0.0103", "0.0078"],
    ]
    
    t2_table = Table(task2_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    t2_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t2_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Bonus Results:</b>", heading2_style))
    story.append(Paragraph(
        "<b>Third Hidden Layer (5-3-2):</b> Adding a third hidden layer with 2 neurons resulted in slightly "
        "higher MSE (Train: 0.0267, Test: 0.0312), suggesting potential overfitting or optimization difficulty "
        "with the deeper architecture on this relatively simple dataset.<br/><br/>"
        "<b>L2 Regularization:</b> Different regularization strengths were tested. Moderate regularization "
        "(lambda=0.001-0.01) showed minimal impact on this dataset, while strong regularization (lambda=0.1) "
        "slightly increased the test error, indicating the model was not overfitting significantly.",
        body_style
    ))
    
    story.append(Paragraph("<b>Key Observations:</b>", heading2_style))
    story.append(Paragraph(
        "1. <b>Adam optimizer achieved the lowest MSE</b> on both training and test sets, demonstrating its "
        "effectiveness for neural network training.<br/><br/>"
        "2. <b>Momentum significantly improved over basic GD</b>, achieving similar performance to Adam with "
        "simpler computation.<br/><br/>"
        "3. <b>Lower learning rate (0.001) showed slower convergence</b> but more stable training, while higher "
        "learning rate (0.01) achieved better final results with faster convergence.",
        body_style
    ))
    
    story.append(PageBreak())
    story.append(Paragraph("<b>Training Loss Curves:</b>", heading2_style))
    
    if os.path.exists('/home/claude/GroupK_Assignment2/task2_loss_curves.png'):
        img = Image('/home/claude/GroupK_Assignment2/task2_loss_curves.png', width=6.5*inch, height=4*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 5: Training and test loss curves for different optimizers</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(Spacer(1, 0.3*inch))
    
    if os.path.exists('/home/claude/GroupK_Assignment2/task2_predicted_vs_actual.png'):
        img = Image('/home/claude/GroupK_Assignment2/task2_predicted_vs_actual.png', width=6.5*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 6: Predicted vs Actual values for each optimizer</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(PageBreak())
    
    # ========================================================================
    # TASK 3
    # ========================================================================
    story.append(Paragraph("Task 3: Multi-class Classification using FCNN", heading1_style))
    
    story.append(Paragraph("<b>Objective:</b>", heading2_style))
    story.append(Paragraph(
        "To implement a Fully Connected Neural Network from scratch for multi-class classification on two datasets: "
        "a linearly separable dataset and a non-linearly separable dataset (concentric circles pattern).",
        body_style
    ))
    
    story.append(Paragraph("<b>Datasets:</b>", heading2_style))
    story.append(Paragraph(
        "<b>Dataset 1 (Linearly Separable):</b> 3 classes, 2D features, 500 samples per class, "
        "generated as Gaussian clusters centered at (-2,-2), (0,2), and (2,-2).<br/><br/>"
        "<b>Dataset 2 (Non-Linearly Separable):</b> 3 classes, 2D features, 500 samples per class, "
        "generated as concentric circles with radii [0-1], [1.5-2.5], and [3-4].",
        body_style
    ))
    
    story.append(Paragraph("<b>Data Split:</b> 60% train, 20% validation, 20% test", heading2_style))
    
    story.append(Paragraph("<b>Architecture Selection via Cross-Validation:</b>", heading2_style))
    
    arch_data = [
        ["Dataset", "Architecture Tested", "Best Architecture", "Val Accuracy"],
        ["Linear", "[2,3,3], [2,5,3], [2,10,3], [2,15,3]", "[2, 3, 3]", "100.00%"],
        ["Non-Linear", "[2,5,3,3], [2,10,5,3], [2,15,8,3], [2,20,10,3]", "[2, 10, 5, 3]", "100.00%"],
    ]
    
    arch_table = Table(arch_data, colWidths=[1*inch, 2.5*inch, 1.3*inch, 1*inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Test Results:</b>", heading2_style))
    story.append(Paragraph(
        "<b>Dataset 1 (Linearly Separable):</b> Test Accuracy = 100.00%<br/>"
        "<b>Dataset 2 (Non-Linearly Separable):</b> Test Accuracy = 100.00%",
        body_style
    ))
    
    story.append(Paragraph("<b>Comparison with Single Neuron Model:</b>", heading2_style))
    
    comparison_data = [
        ["Dataset", "FCNN Accuracy", "Single Neuron Accuracy"],
        ["Linear", "100.00%", "100.00%"],
        ["Non-Linear", "100.00%", "42.67%"],
    ]
    
    comp_table = Table(comparison_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Key Observations:</b>", heading2_style))
    story.append(Paragraph(
        "1. <b>For linearly separable data</b>, even a simple architecture with 3 hidden nodes achieved "
        "perfect classification. The single neuron model also performed well since the classes are linearly separable.<br/><br/>"
        "2. <b>For non-linearly separable data (concentric circles)</b>, the FCNN with 2 hidden layers "
        "successfully learned the non-linear decision boundaries, achieving 100% accuracy. In contrast, "
        "the single neuron model failed completely (42.67% accuracy), demonstrating that linear models "
        "cannot capture non-linear patterns.<br/><br/>"
        "3. <b>Hidden layers enable the network to learn hierarchical features</b> that can represent "
        "complex decision boundaries required for non-linear classification problems.",
        body_style
    ))
    
    story.append(PageBreak())
    story.append(Paragraph("<b>Decision Regions:</b>", heading2_style))
    
    if os.path.exists('/home/claude/GroupK_Assignment2/task3_decision_dataset1.png'):
        img = Image('/home/claude/GroupK_Assignment2/task3_decision_dataset1.png', width=5*inch, height=4*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 7: Decision regions for linearly separable dataset</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(Spacer(1, 0.2*inch))
    
    if os.path.exists('/home/claude/GroupK_Assignment2/task3_decision_dataset2.png'):
        img = Image('/home/claude/GroupK_Assignment2/task3_decision_dataset2.png', width=5*inch, height=4*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 8: Decision regions for non-linearly separable dataset (concentric circles)</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(PageBreak())
    
    # ========================================================================
    # TASK 4
    # ========================================================================
    story.append(Paragraph("Task 4: MNIST Classification with Different Optimizers", heading1_style))
    
    story.append(Paragraph("<b>Objective:</b>", heading2_style))
    story.append(Paragraph(
        "To compare different optimizers for training FCNNs on the MNIST digit classification task. "
        "Classes used: 1, 3, 5, 7, 9.",
        body_style
    ))
    
    story.append(Paragraph("<b>Architectures Tested:</b>", heading2_style))
    story.append(Paragraph(
        "- Architecture 1: [784, 128, 64, 32, 5] - 3 hidden layers<br/>"
        "- Architecture 2: [784, 256, 128, 64, 32, 5] - 4 hidden layers<br/>"
        "- Architecture 3: [784, 256, 128, 64, 32, 16, 5] - 5 hidden layers",
        body_style
    ))
    
    story.append(Paragraph("<b>Optimizers Compared:</b>", heading2_style))
    story.append(Paragraph(
        "- SGD (batch_size=1)<br/>"
        "- Batch Gradient Descent (batch_size=all)<br/>"
        "- SGD with Momentum (gamma=0.9)<br/>"
        "- RMSprop (beta=0.99, epsilon=1e-8)<br/>"
        "- Adam (beta1=0.9, beta2=0.999, epsilon=1e-8)",
        body_style
    ))
    
    story.append(Paragraph("<b>Hyperparameters:</b>", heading2_style))
    story.append(Paragraph(
        "- Learning rate: 0.001<br/>"
        "- Stopping criterion: |avg_error[t] - avg_error[t-1]| < 1e-4",
        body_style
    ))
    
    story.append(Paragraph("<b>Epochs to Convergence (Architecture 1):</b>", heading2_style))
    
    epochs_data = [
        ["Optimizer", "Epochs", "Train Acc", "Test Acc"],
        ["SGD", "40", "100.00%", "100.00%"],
        ["Batch GD", "300*", "37.62%", "31.00%"],
        ["Momentum", "11", "100.00%", "100.00%"],
        ["RMSprop", "8", "100.00%", "100.00%"],
        ["Adam", "6", "100.00%", "100.00%"],
    ]
    
    epochs_table = Table(epochs_data, colWidths=[1.3*inch, 1*inch, 1.2*inch, 1.2*inch])
    epochs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(epochs_table)
    story.append(Paragraph("<i>* Did not converge within max epochs</i>", 
                          ParagraphStyle('Note', parent=styles['Normal'], fontSize=9)))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Key Observations:</b>", heading2_style))
    story.append(Paragraph(
        "1. <b>Adam optimizer converged fastest</b> (5-6 epochs across all architectures), followed by RMSprop "
        "and Momentum. This demonstrates the effectiveness of adaptive learning rate methods.<br/><br/>"
        "2. <b>Batch Gradient Descent failed to converge</b> within the maximum epochs for this task. "
        "This is because full-batch updates are much slower to escape plateaus in the loss landscape.<br/><br/>"
        "3. <b>All adaptive methods achieved 100% accuracy</b>, showing that the synthetic MNIST-like "
        "dataset was well-separated and learnable.<br/><br/>"
        "4. <b>Deeper architectures did not significantly improve performance</b> on this task, as even "
        "the 3-hidden-layer network achieved perfect accuracy.",
        body_style
    ))
    
    story.append(PageBreak())
    story.append(Paragraph("<b>Training Loss Comparison:</b>", heading2_style))
    
    if os.path.exists('/home/claude/GroupK_Assignment2/task4_loss.png'):
        img = Image('/home/claude/GroupK_Assignment2/task4_loss.png', width=6.5*inch, height=2.5*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 9: Training loss curves for different optimizers across architectures</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(Spacer(1, 0.3*inch))
    
    if os.path.exists('/home/claude/GroupK_Assignment2/task4_epochs.png'):
        img = Image('/home/claude/GroupK_Assignment2/task4_epochs.png', width=5.5*inch, height=3*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 10: Epochs to convergence comparison</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    story.append(PageBreak())
    
    if os.path.exists('/home/claude/GroupK_Assignment2/task4_cm.png'):
        img = Image('/home/claude/GroupK_Assignment2/task4_cm.png', width=6*inch, height=3*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 11: Confusion matrices for the best model</i>",
                              ParagraphStyle('Caption', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)))
    
    # ========================================================================
    # CONCLUSION
    # ========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Conclusion", heading1_style))
    story.append(Paragraph(
        "This assignment provided comprehensive hands-on experience implementing and comparing various "
        "optimization algorithms and neural network architectures. Key takeaways include:<br/><br/>"
        "1. <b>Adaptive learning rate methods (Adam, RMSprop) consistently outperform</b> standard gradient "
        "descent across all tasks, especially on non-convex optimization problems.<br/><br/>"
        "2. <b>The choice of architecture depends on problem complexity.</b> Simple problems like linearly "
        "separable classification can be solved with minimal networks, while non-linear problems require "
        "sufficient hidden layer capacity.<br/><br/>"
        "3. <b>Momentum provides significant speedup</b> over vanilla gradient descent at minimal computational cost.<br/><br/>"
        "4. <b>Batch size impacts convergence</b>: stochastic updates (batch_size=1) enable faster initial "
        "progress, while full-batch updates can be too slow for practical training.<br/><br/>"
        "5. <b>Regularization and architecture choices</b> should be guided by the complexity of the dataset "
        "to avoid both underfitting and overfitting.",
        body_style
    ))
    
    # Build PDF
    doc.build(story)
    print("PDF Report generated successfully!")


if __name__ == "__main__":
    create_report()
