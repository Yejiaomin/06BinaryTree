# 06BinaryTree

二叉树的遍历：

递归三要素：
1. 确定递归函数的参数和返回值：List<Integer> inorderTraversal(TreeNode root)
2. 确定终止条件（一般都是root == null）：if (root == null)  return;
3. 确定单层递归的逻辑(前中序遍历想好）：inorder(root.left, list);          list.add(root.val);         inorder(root.right, list);

用stack迭代遍历：
1.前序遍历（中左右）先push(root) -> pop(root) -> push(root.right) -> push(root.left) -> pop(root.left) -> push(root.left.right) -> push(root.left.left)
class Solution {
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            result.add(node.val);
            if (node.right != null){
                stack.push(node.right);
            }
            if (node.left != null){
                stack.push(node.left);
            }
        }
        return result;
    }
}
2.中序遍历（左中右）：入栈顺序： 左-右

class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while (cur != null || !stack.isEmpty()){
           if (cur != null){//不管传入的是左孩子还是右孩子，都去查他的左边。
               stack.push(cur);
               cur = cur.left;//左边全部入完
           }else{
               cur = stack.pop();//左边存完了开始pop（）一个root出来，然后查root的左边。
               result.add(cur.val);
               cur = cur.right;//左边已经没有，查一下右边，进入下一层循环。
           }
        }
        return result;
    }
}
3.后序遍历（左右中）： 前序“中左右”在push的时候改为先左后右，得到“中右左”，然后reverse得到“左右中”。
class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null){
            return result;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()){
            TreeNode node = stack.pop();
            result.add(node.val);
            if (node.left != null){
                stack.push(node.left);
            }
            if (node.right != null){
                stack.push(node.right);
            }
        }
        Collections.reverse(result);
        return result;
    }
}

层序遍历：二叉树关于以层为基础的题目都适用，可用队列先进先出，符合一层一层遍历的逻辑。
class Solution {
    public void checkFun02(TreeNode node) {
        if (node == null) return;
        Queue<TreeNode> que = new LinkedList<TreeNode>();
        que.offer(node);

        while (!que.isEmpty()) {
            List<Integer> itemList = new ArrayList<Integer>();
            int len = que.size();

            while (len > 0) {
                TreeNode tmpNode = que.poll();
                itemList.add(tmpNode.val);

                if (tmpNode.left != null) que.offer(tmpNode.left);
                if (tmpNode.right != null) que.offer(tmpNode.right);
                len--;
            }

            resList.add(itemList);
        }
    }
}
