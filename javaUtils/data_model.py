# @Time : 2021/3/15 14:27
# @Author : Cheng Zhu
# @site : https://gitee.com/lonekey
# @File : data_model.py
import os
import threading
from pprint import pprint
from typing import List, Dict
from javaUtils.java_parse import get_method, java_to_json


def get_key(project: str, kind: str):
    """
    update key of a project
    :param kind:
    :param project:
    :return: new key of a project file
    """
    if not os.path.exists(f'bug_info/{project}/key_{project}_{kind}.txt'):
        with open(f'bug_info/{project}/key_{project}_{kind}.txt', 'w') as f:
            f.write("0")
        f.close()
    with open(f'bug_info/{project}/key_{project}_{kind}.txt', 'r') as f:
        old_key = f.read()
    f.close()
    with open(f'bug_info/{project}/key_{project}_{kind}.txt', 'w') as f:
        new_key = int(old_key) + 1
        f.write(str(new_key))
    f.close()
    return new_key


def split_description(filename):
    with open(filename, 'r', errors='ignore') as f:
        lines = f.readlines()
        f.close()
        # 分离code和description
        descriptions = []
        codes = []
        for line in lines:
            line = line.lstrip().replace('\n', '')
            if line == '':
                continue
            if line.find('/*') >= 0:
                descriptions.append(line)
                continue
            if line.startswith('*'):
                descriptions.append(line)
                continue
            if line.find('*/') >= 0:
                descriptions.append(line)
                continue
            if line.startswith('//'):
                descriptions.append(line)
                continue
            else:
                codes.append(line)
    return descriptions, codes


class IdManager:
    def __init__(self):
        self.id_file: dict = {}
        self.id_method: dict = {}

    def get_id(self, kind, name):
        if kind == "F":
            my_dict = self.id_file
        elif kind == "M":
            my_dict = self.id_method
        else:
            print("type error")
            return
        if name in my_dict.keys():
            file_number, version_number = my_dict[name].split('.')
            my_dict[name] = file_number + '.' + str(int(version_number) + 1)
        else:
            file_number = str(len(my_dict))
            version_number = str(0)
            my_dict[name] = file_number + '.' + version_number
        return my_dict[name]


class Project:
    def __init__(self, project_name, project_path):
        self.project_name = project_name
        self.project_path = project_path
        self.commits: Dict[str, Commit] = {}
        self.bugs: Dict[str, Bug] = {}
        self.id_manager = IdManager()

    def __repr__(self):
        return f"{self.project_name, self.project_path, self.commits, self.bugs}"


class Commit:
    def __init__(self, project: Project, commit_id: str, commit_time: str, fixed_bug_id):
        self.project: Project = project
        self.commit_id: str = commit_id
        self.commit_time: str = commit_time
        self.fixed_bug_id = fixed_bug_id
        self.project_files: List[File] = []

    def __repr__(self):
        # return f"{self.commit_id, self.commit_time, len(self.project_files), [len(i.method_list) for i in
        # self.project_files]}"
        return f"{self.commit_id}"


class Bug:
    def __init__(self, project: Project, bug_id: str, bug_exist_version: Commit, fixed_version: Commit,
                 bug_summary: str,
                 bug_description: str):
        self.project: Project = project
        self.bug_id: str = bug_id
        self.bug_exist_version: Commit = bug_exist_version
        self.fixed_version: Commit = fixed_version
        self.bug_summary = bug_summary
        self.bug_description = bug_description
        self.files: List[File] = []
        self.methods: List[Method] = []
        # 构建bug对象 temp_commit即是before commit, commit对应的即是fixed commit
        # new_bug = Bug(project, k, temp_commit, commit, v["bug_summary"], v["bug_description"])
        # for bf in buggy_files:
        #     new_bug.files.append(copy.copy(bf))
        # for bm in buggy_methods:
        #     new_bug.methods.append(copy.copy(bm))
        # project.bugs[v["bug_id"]] = new_bug  # 将bug与project关联起来
        # print(f"bug {bug_id} finished")

    def __repr__(self):
        return f"{self.bug_id}"


class File:
    def __init__(self, project: Project, commit: Commit, filename: str, id_manager: IdManager, lo: threading.Lock):
        self.project = project
        # 记录创建该文件的提交
        self.create_commit: Commit = commit
        self.filename = filename
        # 记录在哪些提交时发生变更，不关注在一次提交中没有变更的文件,第一个元素就是首次提交, 最后一个元素就是最近提交
        self.change_history: List[Commit] = []
        # 记录删除该文件的提交， create_commit 和 delete_commit之间就是文件存在历史
        self.delete_commit: List[Commit] = []
        self.method_list: List[Method] = []
        lo.acquire()
        file_id = id_manager.get_id("F", filename)
        lo.release()
        self.id: str = file_id
        # self.description, self.code = split_description(filename)
        # rf = java_to_json(self.code, 0, 0)
        ml = get_method(filename)
        if ml is None:
            ml = java_to_json(split_description(filename)[1], 0, 0)["method_list"]
        for me in ml:
            method = Method(self, me, id_manager, lo)
            self.method_list.append(method)
        # print("create a file! ", file_id, filename)

    def __repr__(self):
        # return f"{self.id, self.filename, len(self.method_list), self.uuid, self.create_commit, self.change_history, self.delete_commit}"
        return f"{self.method_list}"


class Method:
    # 每次创建新方法时都分配一个key, 认为方法一旦被修改就是全新的方法,方法没有引起变更的提交历史，只能从头创建，一旦创建属性不可修改
    def __init__(self, file: File, f: dict, id_allow: IdManager, lo: threading.Lock):
        self.method_name = f["name"]
        self.content = f["content"]
        self.comment = f["comment"]
        self.start = f["start"]
        self.end = f["end"]
        self.file: File = file
        # 如果这个方法发生修改了，值设置成False
        self.status = True
        lo.acquire()
        self.id: str = id_allow.get_id("M", self.file.filename + '#' + self.method_name)
        lo.release()

    def __repr__(self):
        return f"{self.id, self.method_name}"


if __name__ == "__main__":
    description, code = split_description("BrowserManager.java")
    pprint(description)
    pprint(code)
    # rf = java_to_json(code, 0, 0)
    # fl = rf["method_list"]
    # pprint(fl)
    # methods = get_method("BrowserManager.java")
    # pprint(methods)
