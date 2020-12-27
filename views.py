from django.shortcuts import render,redirect
from .models import*
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from .forms import *
from django.contrib import messages
from django.contrib.auth.decorators import login_required


# Create your views here.
def index(request):
    form = CreateUserForm()
    print( 'htmlsubmitbutton2' in request.POST)
    if request.method == 'POST' and 'htmlsubmitbutton1' in request.POST:
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request,username = username, password = password)
        if user is not None:
            login(request,user)
            return redirect('profile')

        else:
            messages.info(request,'username or password incorrect.')

    if request.method == 'POST' and 'htmlsubmitbutton2' in request.POST:
        form = CreateUserForm(request.POST)
        print(form)
        if form.is_valid():
            user=form.save()
            username = form.cleaned_data.get('username')
            messages.success(request,'Account was created for ' + username)
            print(user,username)
            return redirect('apply')

    context={'form':form}
    return render(request, 'base/index.html',context)


def About(request):
    return render(request, 'base/aboutus.html')


def Quizz(request):
    if request.method == 'POST':
        form = QuForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('quizz')
            messages.success(request,f'your submission done')
    else:
        form = QuForm()
    return render(request,'base/quizz.html',{'form':form})



def Contact(request):
    form=MessageForm()
    if request.method == 'POST':
        form=MessageForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, f'Your message has been sent')

    context={'form':form}
    return render(request, 'base/contactus.html',context)

def Apply(request):
    if request.method == 'POST':
        form = ApplyForm(request.POST)
        p_form = UserUpdateForm(request.POST)
        if form.is_valid():
            form.save()
            #p_form.save()
            return redirect('index')

    else:
        form = ApplyForm()
        p_form = UserUpdateForm()
    context={'form':form,'p_form':p_form}
    return render(request,'base/apply.html',context)


@login_required(login_url='login')
def Profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST,
                                   request.FILES,
                                   instance=request.user.profile)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, f'Your account has been updated!')
            return redirect('profile')

    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'u_form': u_form,
        'p_form': p_form
    }

    return render(request, 'base/profile.html', context)






def userlogout(request):
    logout(request)
    return redirect('index')
