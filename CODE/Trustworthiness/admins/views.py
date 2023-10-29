from django.shortcuts import render, HttpResponse
from django.contrib import messages
from users.models import UserRegistrationModel
import pandas as pd


# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/viewregisterusers.html', {'data': data})


def adminJruvikaFNDML(request):
    from users.utility import JruvikaMLEDA
    svm_acc, svm_report = JruvikaMLEDA.process_SVM()
    svm_report = pd.DataFrame(svm_report).transpose()
    svm_report = pd.DataFrame(svm_report)
    lg_acc, lg_report = JruvikaMLEDA.process_LogisticRegression()
    lg_report = pd.DataFrame(lg_report).transpose()
    lg_report = pd.DataFrame(lg_report)
    rf_acc, rf_report = JruvikaMLEDA.process_randomForest()
    rf_report = pd.DataFrame(rf_report).transpose()
    rf_report = pd.DataFrame(rf_report)
    nb_acc, nb_report = JruvikaMLEDA.process_naiveBayes()
    nb_report = pd.DataFrame(nb_report).transpose()
    nb_report = pd.DataFrame(nb_report)
    knn_acc, knn_report = JruvikaMLEDA.process_knn()
    knn_report = pd.DataFrame(knn_report).transpose()
    knn_report = pd.DataFrame(knn_report)

    return render(request, 'admins/jrufndml.html',
                  {
                      'svm_report': svm_report.to_html, 'svm_acc': svm_acc,
                      'lg_report': lg_report.to_html, 'lg_acc': lg_acc,
                      'rf_report': rf_report.to_html, 'rf_acc': rf_acc,
                      'nb_report': nb_report.to_html, 'nb_acc': nb_acc,
                      'knn_report': knn_report.to_html, 'knn_acc': knn_acc,
                  })


def adminRealorFakeML(request):
    from users.utility import ReaorFakeML
    results = ReaorFakeML.proces_real_or_fake_dataset()
    return render(request, 'admins/RealorFakeML.html', {'data': results})
