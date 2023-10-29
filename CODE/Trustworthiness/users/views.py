# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def JruvikaDatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'jruvika.csv'
    df = pd.read_csv(path)
    df = df.tail(200)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})


def RealorFakeDatasetView(request):
    fakeNews = settings.MEDIA_ROOT + "//" + 'FakeNews.csv'
    realNews = settings.MEDIA_ROOT + "//" + 'RealNews.csv'
    fakeN = pd.read_csv(fakeNews, nrows=1000)
    realN = pd.read_csv(realNews, nrows=1000)
    df = pd.concat([fakeN, realN], axis="columns")
    df = df.to_html
    return render(request, 'users/realorfake.html', {'data': df})


def usrJruvikaFNDML(request):
    from .utility import JruvikaMLEDA
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

    return render(request, 'users/jruvikaMl.html',
                  {
                      'svm_report': svm_report.to_html, 'svm_acc': svm_acc,
                      'lg_report': lg_report.to_html, 'lg_acc': lg_acc,
                      'rf_report': rf_report.to_html, 'rf_acc': rf_acc,
                      'nb_report': nb_report.to_html, 'nb_acc': nb_acc,
                      'knn_report': knn_report.to_html, 'knn_acc': knn_acc,
                  })


def usrRealorFakeML(request):
    from .utility import ReaorFakeML
    results = ReaorFakeML.proces_real_or_fake_dataset()
    return render(request, 'users/usrRealorFakeML.html', {'data': results})


def predictTrustWorthy(request):
    if request.method == 'POST':
        news  = request.POST.get('news')
        print(news)
        from .utility import JruvikaMLEDA
        result = JruvikaMLEDA.fake_news_det(news)
        return render(request, 'users/testform.html', {'msg': result})
    else:
        return render(request, 'users/testform.html', {})
